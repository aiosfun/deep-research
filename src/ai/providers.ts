import { createGoogleGenerativeAI } from '@ai-sdk/google';
import { createOpenAI } from '@ai-sdk/openai';
import { LanguageModelV1Message } from '@ai-sdk/provider';
import { getEncoding } from 'js-tiktoken';
import { RecursiveCharacterTextSplitter } from './text-splitter';

// 初始化 OpenAI 客户端
const openai = process.env.OPENAI_API_KEY
  ? createOpenAI({
      apiKey: process.env.OPENAI_API_KEY,
      baseURL: process.env.OPENAI_BASE_URL || 'https://api.openai.com/v1',
    })
  : undefined;

// 初始化 Gemini 客户端
const gemini = process.env.GEMINI_API_KEY
  ? createGoogleGenerativeAI({
      apiKey: process.env.GEMINI_API_KEY,
    })
  : undefined;

// 默认模型
const defaultOpenAIModel = openai?.(process.env.OPENAI_MODEL || 'gpt-3.5-turbo');
const defaultGeminiModel = gemini?.(process.env.GEMINI_MODEL || 'gemini-2.0-flash');

export function getModel() {
  const provider = process.env.LLM_PROVIDER || 'gemini';
  
  if (provider === 'openai' && defaultOpenAIModel) {
    return defaultOpenAIModel;
  }
  
  if (provider === 'gemini' && defaultGeminiModel) {
    return defaultGeminiModel;
  }
  
  if (defaultGeminiModel) {
    return defaultGeminiModel;
  }
  
  if (defaultOpenAIModel) {
    return defaultOpenAIModel;
  }
  
  throw new Error('No model initialized. Please check your API keys.');
}

export async function generateText(options: {
  prompt: string;
  system?: string;
}) {
  const model = getModel();
  const formattedPrompt = options.system 
    ? `${options.system}\n\n${options.prompt}`
    : options.prompt;

  const response = await model.doGenerate({
    inputFormat: 'messages',
    mode: { type: 'regular' },
    prompt: [{ role: 'user', content: [{ type: 'text', text: formattedPrompt }] }],
  });
  
  return response.text || '';
}

function cleanJsonString(text: string): string {
  // 移除所有可能的 markdown 标记
  text = text.replace(/```json\s*/g, '').replace(/```\s*$/g, '');
  
  // 移除开头的空白字符和换行
  text = text.trim();
  
  // 如果文本以逗号结尾，移除它
  text = text.replace(/,\s*$/, '');
  
  // 确保对象属性名使用双引号
  text = text.replace(/(\w+):/g, '"$1":');
  
  // 修复可能的转义问题
  text = text.replace(/\\"/g, '"').replace(/\\\\/g, '\\');
  
  return text;
}

export async function generateObject<T>(options: {
  prompt: string;
  system?: string;
  schema: any;
}): Promise<T> {
  const schemaExample = {
    example: "This is an example of the expected format",
    properties: Object.keys(options.schema.shape || {}).reduce((acc, key) => {
      acc[key] = `(${options.schema.shape[key].description || 'required'})`;
      return acc;
    }, {} as Record<string, string>)
  };

  const jsonPrompt = `${options.prompt}

IMPORTANT: You must respond with ONLY a valid JSON object. Follow these rules strictly:
1. Do not include any explanation, markdown formatting, or additional text
2. The response must be a single JSON object
3. All property names must be in double quotes
4. Use the exact property names as specified
5. The response must match this schema exactly:

${JSON.stringify(schemaExample, null, 2)}

Remember: Return ONLY the JSON object, nothing else.`;
  
  const text = await generateText({
    ...options,
    prompt: jsonPrompt,
  });

  let cleanedText = '';
  try {
    cleanedText = cleanJsonString(text);
    console.log('Cleaned JSON text:', cleanedText);
    return JSON.parse(cleanedText) as T;
  } catch (error) {
    console.error('Raw response:', text);
    console.error('Cleaned response:', cleanedText);
    throw new Error(`Failed to parse response as JSON: ${error}`);
  }
}

const MinChunkSize = 140;
const encoder = getEncoding('o200k_base');

// trim prompt to maximum context size
export function trimPrompt(
  prompt: string,
  contextSize = Number(process.env.CONTEXT_SIZE) || 128_000,
) {
  if (!prompt) {
    return '';
  }

  const length = encoder.encode(prompt).length;
  if (length <= contextSize) {
    return prompt;
  }

  const overflowTokens = length - contextSize;
  const chunkSize = prompt.length - overflowTokens * 3;
  if (chunkSize < MinChunkSize) {
    return prompt.slice(0, MinChunkSize);
  }

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize,
    chunkOverlap: 0,
  });
  const trimmedPrompt = splitter.splitText(prompt)[0] ?? '';

  if (trimmedPrompt.length === prompt.length) {
    return trimPrompt(prompt.slice(0, chunkSize), contextSize);
  }

  return trimPrompt(trimmedPrompt, contextSize);
}
