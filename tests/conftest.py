"""pytest conftest - ROS2 launch_testing 플러그인 호환성 문제 우회."""

import pluggy


def pytest_configure(config):
    """비호환 ROS2 pytest 플러그인을 강제 비활성화."""
    pm = config.pluginmanager

    for plugin_name in ["launch_testing", "launch_testing_ros"]:
        plugin = pm.get_plugin(plugin_name)
        if plugin is not None:
            pm.unregister(plugin)

    # launch_testing_ros entrypoint hook도 제거
    for name in list(pm.list_name_plugin()):
        plugin_id, plugin_obj = name
        if "launch_testing" in str(plugin_id):
            try:
                pm.unregister(plugin_obj, plugin_id)
            except Exception:
                pass
