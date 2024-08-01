from PyQt5.QtGui import QIntValidator, QDoubleValidator


def validate_input(lineEdit):
    validator = QDoubleValidator()
    lineEdit.setValidator(validator)
