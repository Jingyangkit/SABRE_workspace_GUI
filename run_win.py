# !/usr/bin/env python
# _*_ coding: utf-8 _*_
import sys
from PyQt5.QtWidgets import QApplication
from ui.windowUI import MainUI

if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = MainUI()
    window.show()
    # window.showMaximized()
    sys.exit(app.exec_())

