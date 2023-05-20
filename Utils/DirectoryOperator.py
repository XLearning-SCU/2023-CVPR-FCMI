import os
import time
import warnings

TestMode = False


class DirectoryOperator:
    def __init__(self, directory: str):
        self.directory = directory

    def make_fold(self):
        if not TestMode:
            # print('mk dir {}'.format(os.path.dirname(self.directory)))
            os.makedirs(os.path.dirname(self.directory), exist_ok=True)

    def modification_time(self):
        if os.path.exists(self.directory):
            return os.path.getmtime(self.directory)
        else:
            warnings.warn('Time_now is returned since the modification time for non-exist file is not available. File: {}'.format(self.directory))
            return time.time()


class FileOperator(DirectoryOperator):
    def __exist(self):
        return os.path.exists(path=self.directory)

    def write(self, data, replace=True, from_begin=False):
        self.make_fold()
        if replace:
            file = open(self.directory, 'w+', encoding='utf-8')
        elif from_begin:
            file = open(self.directory, 'loss+', encoding='utf-8')
        else:
            file = open(self.directory, 'a+', encoding='utf-8')
        if not TestMode:
            file.write(data)
        file.close()

    def read(self, encoding='utf-8'):
        file = open(self.directory, 'r+', encoding=encoding)
        res = file.read()
        file.close()
        return res

    @staticmethod
    def next_filename(dst_file):
        import Utils.ReOperator as ReOperator
        origin_name = FileOperator(directory=dst_file).origin_name()
        extend_name = FileOperator(directory=dst_file).extend_name()
        match_object = ReOperator.ReOperator(string=origin_name).search(pattern=r'_\d*$')
        if match_object is None:
            return os.path.join(os.path.dirname(dst_file), origin_name + '_02' + extend_name)
        else:
            origin_name_2 = ReOperator.ReOperator(string=origin_name).sub(
                pattern=r'_(\d*)$',
                repl=ReOperator.ReplInterface(func=ReOperator.ReOperator.add1).repl)
            return os.path.join(os.path.dirname(dst_file), origin_name_2 + extend_name)

    def rename(self, dst_file, auto_rename=False):
        print('Renaming {} to {}'.format(self.directory, dst_file))
        while auto_rename and os.path.exists(dst_file):
            dst_file = self.next_filename(dst_file=dst_file)
        FileOperator(directory=dst_file).make_fold()
        assert not os.path.exists(dst_file)
        if not TestMode:
            os.rename(src=self.directory, dst=dst_file)

    def copy(self, dst_file, exist_ok=True):
        FileOperator(directory=dst_file).make_fold()
        if exist_ok is False and os.path.exists(dst_file):
            raise OSError
        import shutil
        if not TestMode:
            shutil.copyfile(self.directory, dst_file)

    def extend_name(self):
        return os.path.splitext(self.directory)[1]

    def origin_name(self):
        return os.path.splitext(os.path.basename(self.directory))[0]

    def remove(self):
        if not TestMode:
            os.remove(self.directory)

    def rename_by_regular_expression(self, pattern: str, replacement, auto_rename=False):
        """
        :param pattern:r'my_pattern \] \. ' necessary regular expression
        :param replacement:'substitute ] . ' never regular expression
        :param auto_rename:
        :return:
        DirectoryOperator(R'D:\Temp\b\a3\b12').rename_by_regular_expression(pattern=r'\d', repl='b', )
        """
        import Utils.ReOperator as ReOperator
        res = ReOperator.ReOperator(string=os.path.basename(self.directory)).sub(pattern=pattern, repl=replacement)
        if res != os.path.basename(self.directory):
            self.rename(dst_file=os.path.join(os.path.dirname(self.directory), res), auto_rename=auto_rename)


class FoldOperator(DirectoryOperator):
    def rename_by_regular_expression(self, pattern: str, replacement,
                                     rename_subdirectory=False, auto_rename=False):
        for file_fold in os.listdir(self.directory):
            if rename_subdirectory and os.path.isdir(os.path.join(self.directory, file_fold)):
                FoldOperator(os.path.join(self.directory, file_fold)).rename_by_regular_expression(
                    pattern=pattern, replacement=replacement,
                    rename_subdirectory=rename_subdirectory, auto_rename=auto_rename)
            FileOperator(os.path.join(self.directory, file_fold)).rename_by_regular_expression(
                pattern=pattern, replacement=replacement, auto_rename=auto_rename)

    def copy_modified_files(self, dst_fold, time):
        """

        :param dst_fold:
        :param time:
        :return:
        FoldOperator('D:/L/Deal/Language/JAVA/gdp-master/').copy_modified_files(
        dst_fold='D:/L/Deal/Language/JAVA/gdp-master_commit/',
        time=FileOperator(directory='D:/L/Deal/Language/JAVA/gdp-master/README.md').modification_time())
        """
        for root, folds, files in os.walk(self.directory, topdown=False):
            for file in files:
                target_path = os.path.join(root, file)
                target = FileOperator(target_path)
                if target.modification_time() > time:
                    target.copy(dst_file=target_path.replace(self.directory, dst_fold), exist_ok=True)

    def clear(self, delete_root=True, not_to_delete_file=False):
        if os.path.exists(self.directory):
            print('Clearing {}'.format(self.directory))
        for root, folds, files in os.walk(self.directory, topdown=False):
            if not not_to_delete_file:
                for file in files:
                    if not TestMode:
                        os.remove(os.path.join(root, file))
            for fold in folds:
                if not TestMode:
                    os.rmdir(os.path.join(root, fold))
        if delete_root:
            if os.path.exists(self.directory):
                if not TestMode:
                    os.rmdir(self.directory)
        else:
            DirectoryOperator(os.path.join(self.directory, '...')).make_fold()

    def copy(self, dst_fold, not_to_delete_file=True):
        FoldOperator(dst_fold).clear(delete_root=True, not_to_delete_file=not_to_delete_file)
        print("Copying '{}' to '{}'".format(self.directory, dst_fold))
        import shutil
        if not TestMode:
            shutil.copytree(self.directory, dst_fold)

    def count_folds(self):
        for root, folds, files in os.walk(self.directory):
            return len(folds)
        return 0
