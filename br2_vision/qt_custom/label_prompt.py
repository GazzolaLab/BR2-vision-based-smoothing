import sys
from typing import List

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QGroupBox,
    QHBoxLayout,
    QListWidget,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class LabelPrompt:
    MATCH_FLAG = Qt.MatchFlag.MatchExactly

    def __init__(
        self, app: QApplication, A: List[str], B: List[str], A_name: str, B_name: str
    ):
        assert len(A) > 0, "List A is empty"
        assert len(B) > 0, "List B is empty"
        self.app = app
        self.A = A
        self.B = B
        self.A_name = A_name
        self.B_name = B_name
        self.selected_A_item = A[0]
        self.selected_B_item = B[0]

        self._okay = True

    @property
    def selected(self):
        return (self.selected_A_item, self.selected_B_item)

    def __call__(self):
        # Create main window
        window = QWidget()
        window.setWindowTitle("Select Label")

        layout = QVBoxLayout()
        window.setLayout(layout)

        # Create list widgets (single selection mode)
        list_A = QListWidget()
        list_A.addItems(self.A)
        list_A.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        list_A.clearSelection()
        if self.selected_A_item:
            item = list_A.findItems(self.selected_A_item, self.MATCH_FLAG)[0]
            item.setSelected(True)

        list_B = QListWidget()
        list_B.addItems(self.B)
        list_B.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        list_B.clearSelection()
        if self.selected_B_item:
            item = list_B.findItems(self.selected_B_item, self.MATCH_FLAG)[0]
            item.setSelected(True)

        # Create group boxes
        group_A = QGroupBox(f"List {self.A_name}")
        layout_A = QVBoxLayout()
        group_A.setLayout(layout_A)
        layout_A.addWidget(list_A)

        group_B = QGroupBox(f"List {self.B_name}")
        layout_B = QVBoxLayout()
        group_B.setLayout(layout_B)
        layout_B.addWidget(list_B)

        layout.addWidget(group_A)
        layout.addWidget(group_B)

        # Create OK button
        ok_button = QPushButton("Select")
        layout.addWidget(ok_button)

        # Create cancel button
        cancel_button = QPushButton("Cancel")
        layout.addWidget(cancel_button)

        # Function to handle button click
        def on_ok_click():
            self.selected_A_item = [item.text() for item in list_A.selectedItems()][0]
            self.selected_B_item = [item.text() for item in list_B.selectedItems()][0]
            window.close()
            # self.app.quit()

        def on_cancel_click():
            self._okay = False
            window.close()
            # self.app.quit()

        # Connect button click event
        ok_button.clicked.connect(on_ok_click)
        cancel_button.clicked.connect(on_cancel_click)

        # Show window
        window.show()
        self.app.exec()
        if self._okay:
            return self.selected
        else:
            return None


if __name__ == "__main__":
    # Example usage:
    app = QApplication(sys.argv)
    A = ["a", "b", "c"]
    B = ["1", "2", "3"]
    prompt = LabelPrompt(app, A, B, "a", "2")
    val = prompt()
    print(val)
    val = prompt()
    print(val)
    val = prompt()
    print(val)
    val = prompt()
    print(val)
    val = prompt()
    print(val)
