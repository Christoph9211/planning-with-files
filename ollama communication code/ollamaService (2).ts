/*
 * ollamaService.ts
 *
 * This module replaces the previous Gemini API integration with a client
 * for local Ollama models.  Ollama exposes a simple HTTP interface on
 * `http://localhost:11434` (by default) that can be used to generate
 * responses and maintain conversational context.  We keep track of the
 * conversation history locally and send it on every request so the
 * underlying model has the necessary context to produce coherent
 * continuations. See the official documentation for more details on
 * available parameters.
 *
 * Environment variables are used to customise the base URL and model:
 *
 *   - `OLLAMA_BASE_URL`: The root URL of the Ollama service. Defaults
 *     to `http://127.0.0.1:11434` if unset.
 *   - `OLLAMA_MODEL`: The model name to use when querying Ollama. This
 *     can be any model installed locally (e.g., `huihui_ai/magistral-abliterated:24b`, `gpt-oss:20b`,
 *     `qwen3:30b`). Defaults to `goekdenizguelmez/JOSIEFIED-Qwen3:8b` if unset.
 */

const SYSTEM_INSTRUCTION = `
  You are a masterful storyteller crafting explicit, taboo-oriented narratives where sexual encounters and dark pleasures take center stage.
  Your role is to guide the player through an adult-themed adventure grounded in plausible real world settings. Avoid stereotypes and clichés;
  **Rules:**
  1. **Describe vividly:** Paint sensory-rich pictures using explicit language that engages all senses, especially erogenous zones. Focus on detailed descriptions of intimacy with an emphasis on taboo themes.

  2. **Be responsive:** React to the player's commands with creative outcomes that embrace dark eroticism. If a command doesn't fit, respond in character (e.g., 'You struggle against the restraints.').

  3. **Maintain consistency:** Track the world state, characters' relationships, and intimate encounters throughout the conversation.

  4. **Concise responses:** Keep responses focused on immediate, taboo environments or encounters with 1-2 paragraphs of explicit content.

  5. **Don't lead the player:** Never suggest available commands. Let them discover possibilities through their actions and your descriptions.

  6. **Control the narrative:** Introduce challenges involving seduction, coercion, and dark pleasure. Include characters who may dominate, manipulate, or force themselves on the player.

  7. **No fourth wall:** Never break character or refer to yourself as an AI or game. You are the world.

  Begin Now`;

// Read configuration from environment variables at build time.  Vite's
// `define` replaces these values with the corresponding variables from
// `.env.local` or defaults.  See `vite.config.ts` for details.
const BASE_URL = process.env.OLLAMA_BASE_URL ?? 'http://127.0.0.1:11434';
const MODEL = process.env.OLLAMA_MODEL ?? 'huihui_ai/qwen3-vl-abliterated:8b-instruct';

export interface OllamaModelInfo {
  name: string;
  model: string;
  digest?: string;
}

/**
 * Retrieve the list of models installed in the local Ollama instance.
 * The API returns metadata, but we expose just the fields the UI uses.
 */
export async function fetchAvailableModels(baseUrl: string = BASE_URL): Promise<OllamaModelInfo[]> {
  const response = await fetch(`${baseUrl}/api/tags`);
  if (!response.ok) {
    throw new Error(`Failed to fetch models: ${response.status} ${response.statusText}`);
  }
  const data = await response.json();
  const models = Array.isArray(data?.models) ? data.models : [];
  return models
    .filter((entry: any) => entry && (entry.name || entry.model))
    .map((entry: any) => ({
      name: entry.name ?? entry.model ?? '',
      model: entry.model ?? entry.name ?? '',
      digest: entry.digest,
    }))
    .sort((a: { name: string; }, b: { name: string; }) => a.name.localeCompare(b.name));
}

export interface OllamaMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

// Hide any inline `<think>...</think>` sections that a model may emit.
function stripThinkingText(text: string): string {
  if (!text) return text;
  // Remove one or more <think>...</think> blocks (non-greedy, multiline-safe).
  return text.replace(/<think>[\s\S]*?<\/think>\s*/gi, '').trim();
}

/**
 * Internal representation of a chat session.  We track the base URL,
 * chosen model and the full list of messages exchanged so far.  Each
 * call to Ollama uses the same session state.
 */
export interface OllamaChatSession {
  baseUrl: string;
  model: string;
  messages: OllamaMessage[];
}

/**
 * Initialise a new adventure.  This function seeds the conversation
 * with the system instruction and a "Begin." message from the user.
 * It returns both the session object used to maintain state and the
 * initial reply from the model.  If the underlying request fails,
 * an exception is thrown.
 */
export async function initGame(selectedModel?: string): Promise<{ chat: OllamaChatSession; initialMessage: string }> {
  // Start conversation with system instruction and a kickoff message.
  const messages: OllamaMessage[] = [
    { role: 'system', content: SYSTEM_INSTRUCTION },
    { role: 'user', content: 'Begin.' },
  ];

  const modelToUse = selectedModel?.trim() ? selectedModel.trim() : MODEL;

  const requestBody = {
    model: modelToUse,
    messages,
    stream: false,
    // Disable Ollama "thinking" so the response doesn't include model reasoning.
    // See https://ollama.com/blog/thinking (API supports a `think` boolean).
    think: false,
    options: { temperature: 0.6 },
  };

  const response = await fetch(`${BASE_URL}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(requestBody),
  });
  if (!response.ok) {
    const errorBody = await response.text().catch(() => '');
    const details = errorBody ? ` - ${errorBody}` : '';
    throw new Error(`Failed to start the adventure: ${response.status} ${response.statusText}${details}`);
  }
  const data = await response.json();
  // Extract the assistant's reply.  The API returns an object with
  // `message` containing a `content` property when streaming is disabled【289201217417682†L186-L206】.
  // Prefer the final answer; ignore any `message.thinking` field and strip inline <think> blocks.
  const assistantReplyRaw: string = data?.message?.content ?? '';
  const assistantReply: string = stripThinkingText(assistantReplyRaw);
  // Record assistant message in the history.
  messages.push({ role: 'assistant', content: assistantReply });
  const chat: OllamaChatSession = { baseUrl: BASE_URL, model: modelToUse, messages };
  return { chat, initialMessage: assistantReply };
}

/**
 * Continue the adventure with a user command.  The new message is added
 * to the session history before being sent to the model.  The
 * assistant's reply is appended to the history and returned.  Errors
 * propagate to the caller for handling.
 */
export async function continueGame(chat: OllamaChatSession, message: string): Promise<string> {
  chat.messages.push({ role: 'user', content: message });
  const requestBody = {
    model: chat.model,
    messages: chat.messages,
    stream: false,
    // Keep thinking disabled for follow-ups as well.
    think: false,
    options: { temperature: 0.6 },
  };
  const response = await fetch(`${chat.baseUrl}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(requestBody),
  });
  if (!response.ok) {
    const errorBody = await response.text().catch(() => '');
    const details = errorBody ? ` - ${errorBody}` : '';
    throw new Error(`Failed to continue the adventure: ${response.status} ${response.statusText}${details}`);
  }
  const data = await response.json();
  const assistantReplyRaw: string = data?.message?.content ?? '';
  const assistantReply: string = stripThinkingText(assistantReplyRaw);
  chat.messages.push({ role: 'assistant', content: assistantReply });
  return assistantReply;
}
