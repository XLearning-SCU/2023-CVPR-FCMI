import os


def main():
    for root, folds, files in os.walk('/xlearning/pengxin/Codes/0127/RunSet-0127-Temp'):
        for fold in sorted(folds):
            print(fold)
        break


if __name__ == '__main__':
    main()
