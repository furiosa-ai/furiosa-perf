from furiosa_perf.utils.collect_env import SystemDetector


def test_system_detector() -> None:
    info = SystemDetector.detect_system_info("npu", "furiosa-llm")
    print(info)


test_system_detector()
