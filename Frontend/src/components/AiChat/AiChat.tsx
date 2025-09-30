/**
 * Demo application that uses the BoschLlmApiService.
 * You can use it as a foundation for your app.
 */
import React, { useEffect, useRef, useState } from 'react';
import { Button } from 'primereact/button';
import { InputTextarea } from 'primereact/inputtextarea';
import { Dropdown } from 'primereact/dropdown';
import { Panel } from 'primereact/panel';
import { BlockUI } from 'primereact/blockui';
import { ProgressSpinner } from 'primereact/progressspinner';
import { ConfirmDialog, confirmDialog } from 'primereact/confirmdialog';
import { boschLlmApiService, Message as ApiMessage } from './bosch-llm-api.service';
import { buildCryptoContext, formatCryptoContext } from '../../api/marketService';
import { shouldSearch, searchSerpApi, buildContext as buildWebContext, formatContextMarkdown } from '../../api/webSearchService';
import './AiChat.css';

export const AiChat: React.FC = () => {
  const [chatHistory, setChatHistory] = useState<ApiMessage[]>([]);
  const [userInput, setUserInput] = useState<string>('');
  const [userInputBlocked, setUserInputBlocked] = useState<boolean>(false);
  const [chatEnabled, setChatEnabled] = useState<boolean>(false);
  const [selectedModel, setSelectedModel] = useState<string>('gemini-2.0-flash-lite');
  const [isTyping, setIsTyping] = useState<boolean>(false);
  const [webSearchEnabled, setWebSearchEnabled] = useState<boolean>(true);

  const chatHistoryRef = useRef<HTMLDivElement>(null);
  const msgRefs = useRef<(HTMLElement | null)[]>([]);

  useEffect(() => {
    setChatEnabled(boschLlmApiService.selectModel(selectedModel));
  }, []);

  useEffect(() => {
    const idx = chatHistory.length - 1;
    if (idx >= 0) {
      requestAnimationFrame(() => {
        msgRefs.current[idx]?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      });
    }
  }, [chatHistory]);

  const sendUserInput = async (): Promise<void> => {
    setUserInputBlocked(true);
    setIsTyping(true);
    if (!chatEnabled) {
      setUserInputBlocked(false);
      setIsTyping(false);
      return;
    }

    const inputObject: ApiMessage = { role: 'user', content: userInput };
    setChatHistory(prev => [...prev, inputObject]);
    setUserInput('');

    try {
      let messagesToSend = [...chatHistory, inputObject];
      
      // Add web search context if enabled
      if (webSearchEnabled && shouldSearch(userInput)) {
        try {
          const cryptoContext = await buildCryptoContext(userInput);
          if (cryptoContext) {
            const contextMessage = formatCryptoContext(cryptoContext);
            const systemContext: ApiMessage = {
              role: 'system',
              content: `Context: ${contextMessage}\n\nUse this live market data to provide accurate, up-to-date information about cryptocurrency prices and market trends.`
            };
            messagesToSend = [systemContext, ...messagesToSend];
          }

          // Generic web search via SerpAPI
          const results = await searchSerpApi(userInput);
          console.log('Web search results:', results); // Debug log
          const ctx = buildWebContext(results, userInput);
          if (ctx) {
            const formatted = formatContextMarkdown(ctx);
            console.log('Web context being sent to LLM:', formatted); // Debug log
            const systemWeb: ApiMessage = {
              role: 'system',
              content: formatted
            };
            messagesToSend = [systemWeb, ...messagesToSend];
          }
        } catch (error) {
          console.warn('Web search failed, continuing without live data:', error);
        }
      }

      const response = await boschLlmApiService.get(messagesToSend);
      processResponse(response);
    } catch (error) {
      console.error('API Error:', error);
      handleApiError(error);
    } finally {
      setIsTyping(false);
    }
  };

  const processResponse = (response: any): void => {
    if (response?.choices?.length > 0) {
      const message = response.choices[0].message;
      setChatHistory(prev => [...prev, message]);
    }
    setUserInputBlocked(false);
  };

  const copyToClipboard = (text: string): void => {
    navigator.clipboard.writeText(text).catch(err => {
      console.error('Could not copy text: ', err);
    });
  };

  const resetChatHistory = (): void => {
    setChatHistory([]);
    setUserInput('');
    msgRefs.current = [];
  };

  const confirmResetChatHistory = (): void => {
    confirmDialog({
      message: 'Do you really want to reset the chat history and so delete all entries?',
      header: 'Confirm chat reset',
      icon: 'cui-icon-warning',
      accept: () => resetChatHistory(),
    });
  };

  const onModelChange = (): void => {
    const previous = boschLlmApiService.getCurrentModel();
    const success = boschLlmApiService.selectModel(selectedModel);
    setChatEnabled(success);
    if (!success) setSelectedModel(previous || '');
  };


  const handleApiError = (error: any): void => {
    setUserInputBlocked(false);
    setIsTyping(false);
    const errorMessage = error?.message || 'An error occurred while communicating with the AI service.';
    setChatHistory(prev => [...prev, { role: 'assistant', content: errorMessage }]);
  };

  const handleKeyPress = (e: React.KeyboardEvent): void => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (userInput.trim()) sendUserInput();
    }
  };

  const formatMessageContent = (content: string): string => {
    // Simple HTML escaping and line breaks only
    return content
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/\n/g, '<br>')
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
  };


  return (
    <div>
      <div className="ai-chat-container">
        <div
          ref={chatHistoryRef}
          className={`ai-chat-history-container ${chatHistory.length === 0 ? 'empty' : ''}`}
        >

          {chatHistory.map((message, index) => (
            <React.Fragment key={(message as any).id ?? index}>
              <span
                className="ai-chat-anchor"
                ref={el => {
                  msgRefs.current[index] = el;
                }}
              />
              <section
                className={`ai-chat-message ${
                  message.role === 'user'
                    ? 'ai-chat-user-message'
                    : message.role === 'assistant'
                    ? 'ai-chat-assistant-message'
                    : 'ai-chat-system-message'
                }`}
              >
                <div className="ai-chat-message-toolbar">
                  <Button
                    icon="pi pi-copy"
                    className="p-button-tertiary p-button-icon-only"
                    onClick={() => copyToClipboard(message.content)}
                    aria-label="Copy message"
                    tooltip="Copy message"
                  />
                </div>
                <div className="ai-chat-message-content">
                  <div className="ai-chat-message-text" dangerouslySetInnerHTML={{ __html: formatMessageContent(message.content) }} />
                  <div className="ai-chat-message-time">
                    {new Date().toLocaleTimeString()}
                  </div>
                </div>
              </section>
            </React.Fragment>
          ))}
          
          {isTyping && (
            <section className="ai-chat-message ai-chat-assistant-message">
              <div className="ai-chat-typing-indicator">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </section>
          )}
        </div>

        <BlockUI blocked={userInputBlocked} template={<ProgressSpinner />}>
          <Panel className="ai-chat-user-input-container">
            <div className="ai-chat-user-input-text">
              <div className="ai-chat-input-grid">
                <div className="ai-chat-input-relative">
                  <Dropdown
                    id="modelSelect"
                    value={selectedModel}
                    options={boschLlmApiService.availableModels}
                    optionLabel="label"
                    optionValue="value"
                    onChange={(e) => {
                      setSelectedModel(e.value);
                      onModelChange();
                    }}
                    placeholder="Select model"
                    className="ai-chat-model-selector"
                  />
                  
                  <InputTextarea
                    className="ai-chat-input"
                    value={userInput}
                    onChange={(e) => setUserInput(e.target.value)}
                    onKeyDown={handleKeyPress}
                    disabled={!chatEnabled || userInputBlocked}
                    placeholder={chatEnabled ? "Ask anything..." : "Please select a model to start chatting..."}
                    rows={1}
                    autoResize
                  />
                  
                  <Button
                    icon="pi pi-send"
                    className="p-button-tertiary p-button-icon-only ai-chat-send-button"
                    onClick={sendUserInput}
                    disabled={!chatEnabled || userInput.trim().length === 0 || userInputBlocked}
                    aria-label="Send"
                    tooltip="Send message"
                  />
                  
                  <Button
                    icon={webSearchEnabled ? "pi pi-globe" : "pi pi-globe"}
                    className={`p-button-tertiary p-button-icon-only ai-chat-web-search-button ${webSearchEnabled ? 'web-search-enabled' : ''}`}
                    onClick={() => setWebSearchEnabled(!webSearchEnabled)}
                    aria-label="Toggle web search"
                    tooltip={webSearchEnabled ? "Web search enabled" : "Web search disabled"}
                  />
                  
                  <Button
                    icon="pi pi-refresh"
                    className="p-button-tertiary p-button-icon-only ai-chat-reset-button"
                    onClick={confirmResetChatHistory}
                    disabled={!chatEnabled}
                    aria-label="Reset chat"
                    tooltip="Reset chat"
                  />
                </div>
              </div>
            </div>
          </Panel>
        </BlockUI>

        <ConfirmDialog />
      </div>
    </div>
  );
};
