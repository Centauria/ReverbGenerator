# -*- coding: utf-8 -*-
import argparse
import os


class Checker:
    def __init__(self, name_left, name_right, left=None, right=None):
        self.left = []
        self.right = []
        self.name_left = name_left
        self.name_right = name_right
        if left is not None:
            for item in left:
                self.add_left(item)
        if right is not None:
            for item in right:
                self.add_right(item)

    def __repr__(self):
        s = f'{self.name_left}  |  {self.name_right}\n'
        for i in range(len(self)):
            s = s + f'{self.left[i]}  |  {self.right[i]}\n'
        return s

    def __len__(self):
        length = len(self.left)
        assert length == len(self.right)
        return length

    def add_left(self, item):
        if item not in self.left:
            if item not in self.right:
                self.left.append(item)
                self.right.append(None)
            else:
                index = self.right.index(item)
                self.left[index] = item

    def add_right(self, item):
        if item not in self.right:
            if item not in self.left:
                self.right.append(item)
                self.left.append(None)
            else:
                index = self.left.index(item)
                self.right[index] = item

    def unbalanced(self):
        left_none_index = [i for i, x in enumerate(self.left) if x is None]
        right_none_index = [i for i, x in enumerate(self.right) if x is None]
        left = []
        right = []
        for i in left_none_index:
            left.append(self.left[i])
            right.append(self.right[i])
        for i in right_none_index:
            left.append(self.left[i])
            right.append(self.right[i])
        return Checker(self.name_left, self.name_right, left, right)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('left')
    parser.add_argument('right')
    args = parser.parse_args()

    ldir = os.listdir(args.left)
    rdir = os.listdir(args.right)

    ldir = map(lambda s: os.path.splitext(s)[0], ldir)
    rdir = map(lambda s: os.path.splitext(s)[0], rdir)

    c = Checker('cfg', 'wav', ldir, rdir)
    print(len(c))
    print(c.unbalanced())
