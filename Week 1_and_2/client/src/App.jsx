import { useEffect, useState } from 'react';
import axios from 'axios';

function App() {
  const [todos, setTodos] = useState([]);
  const [text, setText] = useState('');

  const fetchTodos = async () => {
    const { data } = await axios.get('/api/todos');
    setTodos(data);
  };

  useEffect(() => {
    fetchTodos();
  }, []);

  const addTodo = async (e) => {
    e.preventDefault();
    if (!text.trim()) return;
    await axios.post('/api/todos', { text });
    setText('');
    fetchTodos();
  };

  const toggleTodo = async (todo) => {
    await axios.put(`/api/todos/${todo._id}`, { completed: !todo.completed });
    fetchTodos();
  };

  const toggleComplete = async (todo, checked) => {
    await axios.put(`/api/todos/${todo._id}`, { completed: checked });
    fetchTodos();
  };

  const [editingId, setEditingId] = useState(null);
  const [editingText, setEditingText] = useState('');

  const startEdit = (todo) => {
    setEditingId(todo._id);
    setEditingText(todo.text);
  };

  const saveEdit = async (e) => {
    e.preventDefault();
    if (!editingText.trim()) return;
    await axios.put(`/api/todos/${editingId}`, { text: editingText });
    setEditingId(null);
    setEditingText('');
    fetchTodos();
  };

  const cancelEdit = () => {
    setEditingId(null);
    setEditingText('');
  };

  const deleteTodo = async (id) => {
    await axios.delete(`/api/todos/${id}`);
    fetchTodos();
  };

  return (
    <div className="app">
      <h1>MERN To-Do List</h1>
      <form onSubmit={addTodo}>
        <input
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Add a task"
        />
        <button type="submit">Add</button>
      </form>

      <ul>
        {todos.map((todo) => (
          <li key={todo._id} className="todo-row">
            <div className="left">
              <input
                type="checkbox"
                checked={!!todo.completed}
                onChange={(e) => toggleComplete(todo, e.target.checked)}
              />

              {editingId === todo._id ? (
                <form onSubmit={saveEdit} className="edit-form">
                  <input
                    className="edit-input"
                    value={editingText}
                    onChange={(e) => setEditingText(e.target.value)}
                    autoFocus
                  />
                  <button type="submit">Save</button>
                  <button type="button" onClick={cancelEdit}>Cancel</button>
                </form>
              ) : (
                <span className={todo.completed ? 'completed' : ''}>{todo.text}</span>
              )}
            </div>

            <div className="actions">
              {editingId !== todo._id && (
                <>
                  <button onClick={() => startEdit(todo)}>Edit</button>
                  <button onClick={() => deleteTodo(todo._id)}>Delete</button>
                </>
              )}
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}

export default App;
