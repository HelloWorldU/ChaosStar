// src/utils/api.ts
import axios from 'axios';

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000',
  timeout: 600000,
});

// 你也可以在此添加请求/响应拦截器
// api.interceptors.request.use(...);
// api.interceptors.response.use(...);

export default api;
