import sqlite3
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk

class FaceDatabaseViewer:
    def __init__(self, root=None):
        self.conn, self.cursor = self.get_db_connection()
        self.root = root if root else tk.Tk()
        self.root.title("Face Database Viewer")
        self.root.geometry("1600x900")
        self.root.configure(bg="#f0f0f0")

        self.font_large = ("Arial", 14)
        self.font_medium = ("Arial", 12)

        self.style = ttk.Style()
        self.style.configure("Treeview", font=self.font_medium, rowheight=30)
        self.style.configure("Treeview.Heading", font=self.font_large)

        self.tree = ttk.Treeview(self.root, columns=("Name",), show="headings")
        self.tree.heading("Name", text="Name")
        self.tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        for col in ("Name",):
            self.tree.column(col, width=200, anchor="center")

        self.btn_delete = tk.Button(self.root, text="Delete Selected", font=self.font_large, command=self.delete_entry)
        self.btn_delete.pack(pady=15)

        self.refresh_table()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def get_db_connection(self):
        """Initialize and return a database connection and cursor."""
        conn = sqlite3.connect('face_database.db')
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS faces
                          (name TEXT PRIMARY KEY, embedding BLOB)''')
        conn.commit()
        return conn, cursor

    def fetch_entries(self):
        """Fetch all names from the faces table."""
        self.cursor.execute("SELECT name FROM faces")
        return self.cursor.fetchall()

    def refresh_table(self):
        """Refresh the Treeview with the current database entries."""
        for row in self.tree.get_children():
            self.tree.delete(row)
        for entry in self.fetch_entries():
            self.tree.insert("", "end", values=entry)

    def delete_entry(self):
        """Delete the selected entry from the database and refresh the table."""
        selected_item = self.tree.selection()
        if not selected_item:
            messagebox.showwarning("Warning", "Please select an entry to delete")
            return
        name = self.tree.item(selected_item, "values")[0]
        self.cursor.execute("DELETE FROM faces WHERE name = ?", (name,))
        self.conn.commit()
        self.refresh_table()
        messagebox.showinfo("Success", f"Deleted entry: {name}")

    def on_closing(self):
        """Handle cleanup when the window is closed."""
        self.conn.close()
        self.root.destroy()

    def run(self):
        """Start the Tkinter main loop."""
        self.root.mainloop()

if __name__ == "__main__":
    viewer = FaceDatabaseViewer()
    viewer.run()