"""Serve the copied original API/UI on CPU; only SDK children own CUDA."""
def create_app(runtime=None):
    from ttd_model_runtime import Runtime
    from ttd_model_runtime.integrations.fastapi import attach
    from .adapter import load_model, completion, cleanup, release
    from . import bridge
    runtime = runtime or Runtime(load_model, completion=completion, cleanup=cleanup,
        release=release, gpu_process=True, execution_timeout=None)
    bridge.configure(runtime)
    from .app_embedding import app
    return attach(app, runtime=runtime)

def main():
    import uvicorn
    uvicorn.run(create_app(), host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
