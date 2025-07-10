// src/App.tsx
import Chat from './components/Chat';
import './App.css';

export default function App() {
  return (
    <div className="app">
      <header className="chat-header">
        <h2>ChaosStar</h2>
      </header>
      <div className="chat-container">
        <Chat />
      </div>
    </div>
  );
}
