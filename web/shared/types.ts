export interface ChatRequest {
    message: string;
    stream: boolean;
    max_steps?: number;
}
  
export interface ReActResult {
  think: string;
  act: string;
}

export interface StepEvent {
  type: "step";
  data: ReActResult;
}

export interface OtherEvent {
  type: Exclude<string, "step">;
  data: unknown;
}

export type ResponseEvent = StepEvent | OtherEvent;

export interface ChatResponse {
  status: string;
  response: ResponseEvent[] | string;
}
