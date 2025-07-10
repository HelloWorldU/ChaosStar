// src/components/Chat.tsx
import './Chat.css';
import { useEffect, useRef, useState, useCallback } from 'react';
import type { StepEvent, OtherEvent } from '@shared/types';
import { v4 as uuidv4 } from 'uuid';

type ResponseEvent = StepEvent | OtherEvent | { type: 'done'; data: '' } | { type: 'pong'; data: '' };

// 消息类型定义
interface ChatMessage {
  id: string;
  type: 'user' | 'bot';
  content: string;
  timestamp: number;
}

// 专用于websocket回复
interface BotMessage extends ChatMessage {
  type: 'bot';
  thinkingContent?: string;
  responseContent?: string;
  isThinkingComplete?: boolean;
  isResponseComplete?: boolean;
}

export default function Chat() {
  const [messages, setMessages] = useState<(ChatMessage | BotMessage)[]>([]);
  const [input, setInput] = useState('');
  const [connected, setConnected] = useState(false);
  const [streamMode, setStreamMode] = useState<boolean>(false);
  const [loading, setLoading] = useState(false);
  const [reconnecting, setReconnecting] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const httpAbortControllerRef = useRef<AbortController | null>(null);
  const connectionIdRef = useRef<string>(uuidv4());
  const reconnectTimer = useRef<number | null>(null);
  const reconnectAttempts = useRef(0);
  const maxReconnectAttempts = 5;
  const pingInterval = useRef<number | null>(null);
  const chatEndRef = useRef<HTMLDivElement>(null);
  const finishedRef = useRef(false);

  // 自动滚动到底部
  const scrollToBottom = useCallback(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);

  const clearTimers = useCallback(() => {
    if (reconnectTimer.current) {
      clearTimeout(reconnectTimer.current);
      reconnectTimer.current = null;
    }
    if (pingInterval.current) {
      clearInterval(pingInterval.current);
      pingInterval.current = null;
    }
  }, []);

  const startPing = useCallback(() => {
    clearTimers();
    pingInterval.current = window.setInterval(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: 'ping' }));
      }
    }, 10000);
  }, [clearTimers]);

  const connect = useCallback(() => {
    if (reconnecting) return;
    
    setReconnecting(true);
    clearTimers();
    
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    const ws = new WebSocket(`ws://127.0.0.1:8000/ws/${connectionIdRef.current}`);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('WebSocket connected');
      setConnected(true);
      setReconnecting(false);
      reconnectAttempts.current = 0;
      startPing();
    };

    ws.onmessage = (e) => {
      console.log('[WS] onmessage – data=', e.data);
      let evt: ResponseEvent;
      try {
        evt = JSON.parse(e.data);
      } catch (error) {
        console.error('Error parsing message:', error);
        return;
      }

      switch (evt.type) {
        case 'step': {
          const { think, act } = (evt.data as { think: string; act: string });
          let text = ""
          if (think && act) {
            text = `${think}\n${act}`;
          } else if (think && act === '') {
            text = `${think}\n`;
          } else if (think === '' && act) {
            text = `${act}\n`;
          }
          const stepMsg: ChatMessage = {
            id: uuidv4(),
            type: 'bot',
            content: text,
            timestamp: Date.now()
          };
          setMessages(msgs => [...msgs, stepMsg]);
          break;
        }

        case 'thinking': {
          const thinkingData = evt.data as string;
          setMessages(msgs => {
            // 如果最后一条是 bot 消息，且思考还未完成，就更新它；否则追加一条新的 bot 消息
            const lastIdx = msgs.length - 1;
            const lastMsg = msgs[lastIdx];
            if (
              lastMsg &&
              lastMsg.type === 'bot' &&
              !(lastMsg as BotMessage).isThinkingComplete
            ) {
              // 更新最后一条
              return msgs.map((msg, i) => {
                if (i !== lastIdx) return msg;
                const prev = msg as BotMessage;
                const newThinking = prev.thinkingContent
                  ? prev.thinkingContent + thinkingData
                  : thinkingData;
                return {
                  ...prev,
                  thinkingContent: newThinking,
                  // 如果你要让 content 同步显示，也可以这样拼接
                  content: newThinking + (prev.responseContent || '')
                };
              });
            } else {
              // 追加新的一条 bot 消息
              const newBotMsg: BotMessage = {
                id: uuidv4(),
                type: 'bot',
                content: thinkingData,
                timestamp: Date.now(),
                thinkingContent: thinkingData,
                responseContent: '',
                isThinkingComplete: false,
                isResponseComplete: false,
              };
              return [...msgs, newBotMsg];
            }
          });
          break;
        }
          
        case 'response': {
          const responseData = evt.data as string;
          setMessages(msgs =>
            msgs.map((msg, i) => {
              if (i === msgs.length - 1 && msg.type === 'bot') {
                const prev = msg as BotMessage;
                const newResp = prev.responseContent
                  ? prev.responseContent + responseData
                  : responseData;
                return {
                  ...prev,
                  responseContent: newResp,
                  content: (prev.thinkingContent || '') + newResp
                };
              }
              return msg;
            })
          );
          break;
        }

        case 'block_stop': {
          setMessages(msgs => {
            const newMsgs = [...msgs];
            const lastMsg = newMsgs[newMsgs.length - 1];
            
            if (lastMsg && lastMsg.type === 'bot') {
              const botMsg = lastMsg as BotMessage;
              // 根据当前状态判断是思考结束还是回复结束
              if (!botMsg.isThinkingComplete && botMsg.thinkingContent) {
                botMsg.isThinkingComplete = true;
              } else if (botMsg.isThinkingComplete && !botMsg.isResponseComplete) {
                botMsg.isResponseComplete = true;
              }
            }
            return newMsgs;
          });
          break;
        }

        case 'done': {
          finishedRef.current = true;
          setLoading(false);
          setMessages(msgs => {
            const newMsgs = [...msgs];
            const lastMsg = newMsgs[newMsgs.length - 1];
            
            if (lastMsg && lastMsg.type === 'bot') {
              const botMsg = lastMsg as BotMessage;
              botMsg.isThinkingComplete = true;
              botMsg.isResponseComplete = true;
            }
            return newMsgs;
          });
          break;
        }

        case 'error': {
          setLoading(false);
          const errorMsg: ChatMessage = {
            id: uuidv4(),
            type: 'bot',
            content: `错误：${evt.data}`,
            timestamp: Date.now()
          };
          setMessages(msgs => [...msgs, errorMsg]);
          break;
        }

        case 'pong':
          console.log('Received pong');
          break;

        default:
          console.log('Unknown message type:', evt);
      }
    };

    ws.onclose = (event) => {
      console.log('WebSocket closed:', event.code, event.reason);
      setConnected(false);
      setReconnecting(false);
      clearTimers();

      if (finishedRef.current) {
        console.log('对话已结束，无需重连');
        return;   // 直接 return，不走重连逻辑
      }

      if (event.code !== 1000 && reconnectAttempts.current < maxReconnectAttempts) {
        reconnectAttempts.current++;
        const delay = Math.min(1000 * Math.pow(2, reconnectAttempts.current), 10000);
        console.log(`Reconnecting in ${delay}ms (attempt ${reconnectAttempts.current})`);
        
        reconnectTimer.current = window.setTimeout(() => {
          connect();
        }, delay);
      } else if (reconnectAttempts.current >= maxReconnectAttempts) {
        console.log('Max reconnection attempts reached');
        const errorMsg: ChatMessage = {
          id: uuidv4(),
          type: 'bot',
          content: '连接失败，请刷新页面重试',
          timestamp: Date.now()
        };
        setMessages(msgs => [...msgs, errorMsg]);
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setReconnecting(false);
    };
  }, [reconnecting, startPing, clearTimers]);

  useEffect(() => {
    if (streamMode) {
      connect();
    }
    return () => {
      wsRef.current?.close();
      wsRef.current = null;
      clearTimers();
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [streamMode]);

  async function sendHttpRequest(message: string): Promise<string> {
    const ctrl = new AbortController();
    httpAbortControllerRef.current = ctrl;
    const resp = await fetch('http://127.0.0.1:8000/chat', {
      method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message,
        stream: false,
      }),
      signal: ctrl.signal,
    });

    // 如果 HTTP 状态不是 200–299，就抛出更详细的错误
    if (!resp.ok) {
      let errBody: unknown;
      try {
        errBody = await resp.json();
      } catch {
        errBody = await resp.text();
      }
      throw new Error(`HTTP ${resp.status} ${resp.statusText}: ${JSON.stringify(errBody)}`);
    }
    const data = await resp.json();
    if (data.status === 'success') {
      type StepResponseItem = { type: 'step'; data: { think: string; act: string } };
      const stepResults = data.response.filter((item: StepResponseItem) => item.type === 'step');
      const responseText = stepResults
        .map((item: StepResponseItem) => {
          const { think, act } = item.data;
          let text = ""
          if (think && act) {
            text = `${think}\n${act}`;
          } else if (think && act === '') {
            text = `${think}\n`;
          } else if (think === '' && act) {
            text = `${act}\n`;
          }
          return text;
        })
        .join('\n');
      return responseText;
    } else {
      throw new Error(data.response || 'HTTP 请求失败');
    }
  }
  
  const send = useCallback(async () => {
    if (!input.trim()) return;

    setLoading(true);
    const userMsg: ChatMessage = {
      id: uuidv4(),
      type: 'user',
      content: input.trim(),
      timestamp: Date.now()
    };

    setMessages(msgs => [...msgs, userMsg]);
    setInput('');

    try {
      if (streamMode) {
        if (wsRef.current.readyState === WebSocket.OPEN) {
          wsRef.current.send(JSON.stringify({ type: 'chat', message: userMsg.content }));
        }
        else {
          console.warn('WebSocket not ready');
          return;
        }
      } else {
        const reply = await sendHttpRequest(userMsg.content)
        const botMsg: ChatMessage = {
          id: uuidv4(),
          type: 'bot',
          content: reply,
          timestamp: Date.now()
        };
        setMessages(msgs => [...msgs, botMsg]);
      }
    } catch (err: unknown) {
      if (err instanceof Error) {
        const msgText = err.name === 'AbortError' ? '用户已停止' : `错误：${err.message}`;
        setMessages(msgs => [
          ...msgs,
          {
            id: uuidv4(),
            type: 'bot',
            content: msgText,
            timestamp: Date.now(),
          }
        ]);
      }
    } finally {
      if (!streamMode) {
        setLoading(false);
      }
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [input, streamMode, wsRef.current]);

  const handleKeyPress = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  }, [send]);
  
  const stopGeneration = useCallback(() => {
    if (streamMode) {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: 'stop' }));
      }
    } else {
      httpAbortControllerRef.current?.abort();
      httpAbortControllerRef.current = null;
      setLoading(false);
    }
  }, [streamMode]);

  const reconnect = useCallback(() => {
    if (!reconnecting) {
      reconnectAttempts.current = 0;
      connectionIdRef.current = uuidv4();
      connect();
    }
  }, [reconnecting, connect]);

  // 渲染消息组件
  const renderMessage = (msg: ChatMessage | BotMessage) => {
    if (msg.type === 'user') {
      return (
        <div key={msg.id} className="message user">
          <div className="message-content user-message">
            {msg.content}
          </div>
        </div>
      );
    }

    const botMsg = msg as BotMessage;
    
    return (
      <div key={msg.id} className="message bot">
        <div className="bot-message-container">
          {/* 思考块 */}
          {botMsg.thinkingContent && (
            <div className={`thinking-block ${botMsg.isThinkingComplete ? 'complete' : 'streaming'}`}>
              <div className="thinking-header">
                <span className="thinking-icon">💭</span>
                <span className="thinking-label">AI 正在思考...</span>
                {botMsg.isThinkingComplete && (
                  <span className="complete-badge">完成</span>
                )}
              </div>
              <div className="thinking-content">
                {botMsg.thinkingContent}
                {!botMsg.isThinkingComplete && (
                  <span className="cursor-blink">|</span>
                )}
              </div>
            </div>
          )}
          
          {/* 回复块 */}
          {botMsg.responseContent && (
            <div className={`response-block ${botMsg.isResponseComplete ? 'complete' : 'streaming'}`}>
              <div className="response-content">
                {botMsg.responseContent}
                {!botMsg.isResponseComplete && (
                  <span className="cursor-blink">|</span>
                )}
              </div>
            </div>
          )}
          
          {/* 如果既没有思考内容也没有回复内容，显示普通消息 */}
          {!botMsg.thinkingContent && !botMsg.responseContent && msg.content && (
            <div
              key={msg.id}
              className={`message bot ${msg.content.includes('停止') ? 'stopped' : ''}`}
            >
              <div className="message-content">{msg.content}</div>
            </div>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="chat-container">
      <div className="chat-window">
        {messages.map(renderMessage)}
        
        {streamMode && reconnecting && (
          <div className="status-message reconnecting">
            <span className="status-icon">🔄</span>
            正在重连... (尝试 {reconnectAttempts.current}/{maxReconnectAttempts})
          </div>
        )}
        
        {streamMode && !connected && !reconnecting && (
          <div className="status-message disconnected">
            <span className="status-icon">❌</span>
            连接已断开 
            <button onClick={reconnect} className="reconnect-btn">
              重新连接
            </button>
          </div>
        )}
        
        <div ref={chatEndRef} />
      </div>
      
      <div className='chat-footer'>
        <div className="chat-input">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            disabled={streamMode ? !connected : false}
            placeholder={streamMode
              ? (connected ? "输入消息..." : "WebSocket连接中...")
              : "输入消息..."}
          />
          <button
            onClick={send}
            disabled={
              input.trim() === '' ||
              (streamMode? !connected : false)
            }
          >
            发送
          </button>
          <button
            onClick={stopGeneration}
            disabled={!loading}
            className="stop-btn"
          >
            停止
          </button>
        </div>

        {/* —— Stream Mode 切换 —— */}
        <div className="stream-toggle">
          <label>
            <input
              type="checkbox"
              checked={streamMode}
              onChange={() => setStreamMode(prev => !prev)}
            />{' '}
            Stream mode
          </label>
        </div>
        <div className="status-bar">
          {streamMode
            ? `状态: ${connected ? '已连接' : reconnecting ? '重连中' : '已断开'}${reconnectAttempts.current > 0 ? ` (重连次数: ${reconnectAttempts.current})` : ''}`
            : '模式: HTTP 非流式'}
        </div>

    </div>
      </div>

  );
}
