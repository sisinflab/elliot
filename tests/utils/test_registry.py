import pytest

from elliot.utils.callback import ElliotCallback
from elliot.utils.registry import BasicRegistry, CallbackRegistry


class TestBasicRegistry:

    def test_register_and_fetch(self):
        registry = BasicRegistry("TestRegistry")

        @registry.register()
        class TestClass:
            def __init__(self, value):
                self.value = value

        instance = registry.get("TestClass", value=42)
        assert instance.value == 42

    def test_register_with_custom_name(self):
        registry = BasicRegistry("TestRegistry")

        @registry.register(name="CustomName")
        class AnotherClass:
            def __init__(self):
                self.data = "test"

        instance = registry.get("CustomName")
        assert instance.data == "test"

    def test_register_and_fetch_multiple_classes(self):
        registry = BasicRegistry("TestRegistry")

        @registry.register()
        class FirstClass:
            pass

        @registry.register()
        class SecondClass:
            pass

        all_classes = registry.all()
        assert "FirstClass" in all_classes
        assert "SecondClass" in all_classes

    def test_register_with_extra_keywords(self):
        registry = BasicRegistry("TestRegistry")

        @registry.register(extra_attribute="example")
        class TestClass:
            pass

        cls = registry.get_class("TestClass")
        assert getattr(cls, "extra_attribute", None) == "example"

    def test_missing_class(self):
        registry = BasicRegistry("TestRegistry")

        with pytest.raises(ValueError):
            registry.get("NonExistentClass")


class TestCallbackRegistry:

    def test_register_and_list_all(self):
        registry = CallbackRegistry("TestRegistry")

        @registry.register()
        class CallbackA(ElliotCallback):
            pass

        @registry.register()
        class CallbackB(ElliotCallback):
            pass

        registered_names = registry.all()
        assert registered_names == ["CallbackA", "CallbackB"]

    def test_register_with_priority(self):
        registry = CallbackRegistry("PriorityRegistry")

        @registry.register(priority=2)
        class CallbackA(ElliotCallback):
            pass

        @registry.register(priority=1)
        class CallbackB(ElliotCallback):
            pass

        @registry.register()
        class CallbackC(ElliotCallback):
            pass  # Default priority should be 3 (max priority + 1)

        @registry.register(priority=5)
        class CallbackD(ElliotCallback):
            pass

        callbacks = registry.get_all()  # Should be sorted by priority (low to high)
        assert (
            [cb.__class__.__name__ for cb in callbacks] ==
            ["CallbackB", "CallbackA", "CallbackC", "CallbackD"]
        )

    def test_register_with_custom_name(self):
        registry = CallbackRegistry("CustomNameRegistry")

        @registry.register(name="CustomName")
        class CallbackA(ElliotCallback):
            pass

        callbacks = registry.all()
        assert callbacks == ["CustomName"]

    def test_register_with_extra_keyword(self):
        registry = CallbackRegistry("MetadataRegistry")

        @registry.register(custom_key="custom_value")
        class CallbackA(ElliotCallback):
            pass

        callbacks = registry.get_all()
        assert len(callbacks) == 1
        callback_instance = callbacks[0]
        assert getattr(callback_instance.__class__, "custom_key", None) == "custom_value"


if __name__ == "__main__":
    pytest.main()
