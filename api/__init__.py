"""API package for the CUDA Agentic Optimizer."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


def create_app() -> FastAPI:
    app = FastAPI(
        title="CUDA Agentic Optimizer API",
        description="Backend API for the optimization frontend",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    from api.problems import router as problems_router
    from api.tree import router as tree_router
    from api.optimizer import router as optimizer_router
    from api.branches import router as branches_router

    app.include_router(problems_router)
    app.include_router(tree_router)
    app.include_router(optimizer_router)
    app.include_router(branches_router)

    @app.on_event("shutdown")
    def _shutdown():
        from api.helpers import terminate_all_optimizers

        terminate_all_optimizers()

    return app
