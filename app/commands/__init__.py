def define_tasks(scheduler, app):
    @scheduler.task(
        "interval",
        id="test_command",
        seconds=10,
        misfire_grace_time=900,
        max_instances=1,
    )
    def test_command():
        print("Test command executed.")
