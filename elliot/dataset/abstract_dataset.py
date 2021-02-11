from abc import abstractmethod


class ForceRequiredAttributeDefinitionMeta(type):

    def __call__(cls, *args, **kwargs):
        class_object = type.__call__(cls, *args, **kwargs)
        cls.check_required_attributes(class_object)
        return class_object

    def check_required_attributes(cls, class_object):
        missing_attrs = [f"{attr}" for attr in class_object.required_attributes
                         if not hasattr(class_object, attr)]
        if missing_attrs:
            raise NotImplementedError("class '%s' requires attribute%s %s" %
                                 (class_object.__class__.__name__, "s" * (len(missing_attrs) > 1),
                                  ", ".join(missing_attrs)))


class AbstractDataset(metaclass=ForceRequiredAttributeDefinitionMeta):

    required_attributes = [
        "config",  # comment
        "args",  # comment
        "kwargs",  # comment
        "users",  # comment
        "items",  # comment
        "num_users",  # comment
        "num_items",  # comment
        "private_users",  # comment
        "public_users",  # comment
        "private_items",  # comment
        "public_items",  # comment
        "transactions",  # comment
        "train_dict",  # comment
        "i_train_dict",  # comment
        "sp_i_train",  # comment
        "test_dict"  # comment
    ]

    @abstractmethod
    def build_dict(self):
        raise NotImplementedError

    @abstractmethod
    def build_sparse(self, *args):
        raise NotImplementedError

    @abstractmethod
    def get_test(self, *args):
        raise NotImplementedError

