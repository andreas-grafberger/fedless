import asyncio
import logging
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import urllib3
from pydantic import ValidationError
from requests import Session

from fedless.benchmark.strategies.base import ServerlessFlStrategy
from fedless.invocation import retry_session, InvocationError
from fedless.models import (
    ClientConfig,
    FunctionDeploymentConfig,
    MongodbConnectionConfig,
    DatasetLoaderConfig,
    FunctionInvocationConfig,
    InvocationResult,
    InvokerParams,
)
from fedless.providers import FaaSProvider

logger = logging.getLogger(__name__)


class FedkeeperStrategy(ServerlessFlStrategy):
    def __init__(
        self,
        provider: FaaSProvider,
        clients: List[ClientConfig],
        invoker_config: FunctionDeploymentConfig,
        mongodb_config: MongodbConnectionConfig,
        evaluator_config: FunctionDeploymentConfig,
        aggregator_config: FunctionDeploymentConfig,
        client_timeout: float = 300,
        global_test_data: Optional[DatasetLoaderConfig] = None,
        aggregator_params: Optional[Dict] = None,
        allowed_stragglers: int = 0,
        use_separate_invokers: bool = True,
        session: Optional[str] = None,
        save_dir: Optional[Path] = None,
        proxies: Dict = None,
        invocation_delay: float = None,
    ):
        super().__init__(
            provider=provider,
            session=session,
            clients=clients,
            mongodb_config=mongodb_config,
            global_test_data=global_test_data,
            aggregator_params=aggregator_params,
            evaluator_config=evaluator_config,
            aggregator_config=aggregator_config,
            client_timeout=client_timeout,
            allowed_stragglers=allowed_stragglers,
            save_dir=save_dir,
            proxies=proxies,
            invocation_delay=invocation_delay,
        )
        self.use_separate_invokers = use_separate_invokers
        self.invoker_config: FunctionDeploymentConfig = invoker_config

        # Will be set during deployment
        self._invoker: Optional[FunctionInvocationConfig] = None
        self._client_to_invoker: Optional[Dict[str, FunctionInvocationConfig]] = None

    @property
    def client_to_invoker(self) -> Dict[str, FunctionInvocationConfig]:
        if self._client_to_invoker is None:
            raise ValueError()
        return self._client_to_invoker

    async def deploy_all_functions(self, *args, **kwargs):
        logger.info(f"Deploying fedkeeper functions...")
        logger.info(f"Deploying aggregator and evaluator")
        self._aggregator = await self.provider.deploy(self.aggregator_config.params)
        self._evaluator = await self.provider.deploy(self.evaluator_config.params)

        if self.use_separate_invokers:
            client_ids = [client.client_id for client in self.clients]
            logger.info(
                f"Found {len(client_ids)} client functions. Deploying one invoker each"
            )
            client_invoker_mappings = []
            for ix, client in enumerate(client_ids):
                invoker_config = self.invoker_config.copy(deep=True)

                invoker_config.params.name = f"{invoker_config.params.name}-{ix}"
                invoker_invocation_config = await self.provider.deploy(
                    invoker_config.params
                )
                client_invoker_mappings.append((client, invoker_invocation_config))
                logger.debug(f"Deployed invoker {invoker_config.params.name}")
            self._client_to_invoker = {c: inv for (c, inv) in client_invoker_mappings}
        else:
            invoker = await self.provider.deploy(self.invoker_config.params)
            self._client_to_invoker = defaultdict(lambda: invoker)
            logger.debug(f"Deployed invoker {invoker.params.name}")

    async def call_clients(
        self, round: int, clients: List[ClientConfig]
    ) -> Tuple[List[InvocationResult], List[str]]:
        urllib3.disable_warnings()
        tasks = []

        for client in clients:
            session = Session()
            # session.proxies.update(self.proxies)
            session = retry_session(session=session)
            params = InvokerParams(
                session_id=self.session,
                round_id=round,
                client_id=client.client_id,
                database=self.mongodb_config,
                http_proxies=self.proxies,
            )
            invoker = self.client_to_invoker[client.client_id]

            # function with closure for easier logging
            async def _inv(function, data, session, round, client_id):
                try:
                    if self.invocation_delay:
                        await asyncio.sleep(random.uniform(0.0, self.invocation_delay))
                    t_start = time.time()
                    res = await self.invoke_async(
                        function, data, session=session, timeout=self.client_timeout
                    )
                    dt_call = time.time() - t_start
                    self.client_timings.append(
                        {
                            "client_id": client_id,
                            "session_id": self.session,
                            "invocation_time": t_start,
                            "function": function.json(),
                            "seconds": dt_call,
                            "round": round,
                        }
                    )
                    return res
                except InvocationError as e:
                    return str(e)

            tasks.append(
                asyncio.create_task(
                    _inv(
                        function=invoker,
                        data=params.dict(),
                        session=session,
                        round=round,
                        client_id=client.client_id,
                    )
                )
            )

        done, pending = await asyncio.wait(tasks)
        results = list(map(lambda f: f.result(), done))
        suc, errs = [], []
        for res in results:
            try:
                suc.append(InvocationResult.parse_obj(res))
            except ValidationError:
                errs.append(res)
        return suc, errs
