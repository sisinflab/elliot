import pytest

from elliot.utils.callback import ElliotCallback, CallbackManager


class TestCallbackManager:

    def test_trigger_existing_event(self):
        class MockCallback(ElliotCallback):
            def __init__(self):
                super().__init__()
                self.event_called = False

            def mock_event(self):
                self.event_called = True

        callback_instance = MockCallback()
        callback_manager = CallbackManager(callbacks=[callback_instance])

        callback_manager.trigger("mock_event")
        assert callback_instance.event_called is True

    def test_trigger_non_existing_event(self):
        class MockCallback(ElliotCallback):
            def __init__(self):
                super().__init__()
                self.event_called = False

        callback_instance = MockCallback()
        callback_manager = CallbackManager(callbacks=[callback_instance])

        callback_manager.trigger("non_existing_event")
        assert callback_instance.event_called is False

    def test_trigger_event_with_arguments(self):
        class MockCallback(ElliotCallback):
            def __init__(self):
                super().__init__()
                self.received_value = None

            def mock_event(self, value):
                self.received_value = value

        callback_instance = MockCallback()
        callback_manager = CallbackManager(callbacks=[callback_instance])

        callback_manager.trigger("mock_event", value=42)
        assert callback_instance.received_value == 42

    def test_trigger_event_with_multiple_callbacks(self):
        call_order = []

        class MockCallbackA(ElliotCallback):
            def __init__(self):
                super().__init__()
                self.event_called = False

            def mock_event(self):
                self.event_called = True
                call_order.append("A")

        class MockCallbackB(ElliotCallback):
            def __init__(self):
                super().__init__()
                self.event_called = False

            def mock_event(self):
                self.event_called = True
                call_order.append("B")

        callback_A = MockCallbackA()
        callback_B = MockCallbackB()
        callback_manager = CallbackManager(callbacks=[callback_A, callback_B])

        callback_manager.trigger("mock_event")
        assert callback_A.event_called is True
        assert callback_B.event_called is True
        assert call_order == ["A", "B"]


if __name__ == '__main__':
    pytest.main()
