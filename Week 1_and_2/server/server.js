import express from 'express';
import cors from 'cors';
import mongoose from 'mongoose';
import dotenv from 'dotenv';

dotenv.config();

const app = express();
const PORT = process.env.PORT || 5000;
const MONGO_URI = process.env.MONGO_URI || 'mongodb://127.0.0.1:27017/todo-mern';

app.use(cors());
app.use(express.json());

const todoSchema = new mongoose.Schema({
  text: { type: String, required: true },
  completed: { type: Boolean, default: false }
}, { timestamps: true });

const Todo = mongoose.model('Todo', todoSchema);
let todosInMemory = [];
let mongoReady = false;

const sortTodos = (items) => items.slice().sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt));

const getTodos = async () => {
  if (mongoReady) {
    return Todo.find().sort({ createdAt: -1 });
  }
  return sortTodos(todosInMemory);
};

const createTodo = async (text) => {
  if (mongoReady) {
    const todo = new Todo({ text });
    await todo.save();
    return todo;
  }

  const todo = {
    _id: String(Date.now()),
    text,
    completed: false,
    createdAt: new Date(),
    updatedAt: new Date()
  };
  todosInMemory.unshift(todo);
  return todo;
};

const updateTodo = async (id, updates) => {
  if (mongoReady) {
    return Todo.findByIdAndUpdate(id, updates, { new: true });
  }

  const todo = todosInMemory.find((item) => item._id === id);
  if (!todo) return null;
  Object.assign(todo, updates, { updatedAt: new Date() });
  return todo;
};

const deleteTodo = async (id) => {
  if (mongoReady) {
    await Todo.findByIdAndDelete(id);
    return;
  }
  todosInMemory = todosInMemory.filter((item) => item._id !== id);
};

app.get('/api/todos', async (_req, res) => {
  res.json(await getTodos());
});

app.post('/api/todos', async (req, res) => {
  const todo = await createTodo(req.body.text);
  res.status(201).json(todo);
});

app.put('/api/todos/:id', async (req, res) => {
  const todo = await updateTodo(req.params.id, req.body);
  res.json(todo);
});

app.delete('/api/todos/:id', async (req, res) => {
  await deleteTodo(req.params.id);
  res.json({ success: true });
});

const connectToDatabase = async () => {
  try {
    await mongoose.connect(MONGO_URI);
    mongoReady = true;
    console.log('MongoDB connected');
  } catch (error) {
    console.warn('MongoDB unavailable, using in-memory storage:', error.message);
  }
};

connectToDatabase().then(() => {
  app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
});
