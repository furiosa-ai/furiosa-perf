BENCH_CONFIG_DIR="./bench"
SERVER_CONFIG_DIR="./server"

furiosa-perf run \
    --backend furiosa-llm \
    --hardware-type npu \
    --server-config $SERVER_CONFIG_DIR/exaone4.yaml \
    --benchmark-config $BENCH_CONFIG_DIR/llm_scenario.yaml \
    --model-id furiosa-ai/EXAONE-4.0-32B-FP8