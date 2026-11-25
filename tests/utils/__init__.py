from elliot.utils.folder import path_joiner
from tests import test_path


reader_folder = path_joiner(test_path, "utils", "reader")
reader_path = lambda file_name=None, ext="tsv", folder=None: (
    path_joiner(reader_folder, folder or f"{file_name}.{ext}")
)

writer_path = lambda folder, file_name=None, ext="tsv", subfolder=None: (
    path_joiner(folder, subfolder or f"{file_name}.{ext}")
)
