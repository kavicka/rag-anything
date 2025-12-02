import React, { useState, useEffect, useRef, useCallback } from 'react'
import { MessageCircle, Upload, Send, FileText, Trash2, CheckCircle, Clock, Play, X, Square, AlertCircle, LogOut, Plus, Edit2 } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypeRaw from 'rehype-raw'
import config, { buildApiUrl, fetchWithAuth } from './config'
import { useAuth } from './auth/AuthContext'
import Login from './auth/Login'

// Types
interface ChatMessage {
    id: string
    content: string
    sender: 'user' | 'assistant'
    timestamp: string
    loading?: boolean
}

interface TableData {
    page: string
    type: string
    chunk: string
    filename?: string
    content?: string
}

interface ParsedMessage {
    text: string
    tables: TableData[]
}

interface Document {
    id: string
    name: string
    path: string
    size: number
    upload_time: string
    processed: boolean
}

interface UploadStep {
    id: string
    name: string
    description: string
    status: 'pending' | 'in_progress' | 'completed' | 'error'
    progress: number
    error?: string
}

interface UploadedFile {
    file: File
    id: string
    status: 'pending' | 'uploading' | 'processing' | 'completed' | 'error'
    progress: number
    error?: string
    steps: UploadStep[]
}

interface ProcessingStatus {
    document_id: string
    status: 'processing' | 'completed' | 'error'
    progress: number
    message: string
}

interface Chat {
    id: string
    user_id: string
    name: string
    is_temporary: boolean
    created_at: string
    updated_at: string
    last_message_at: string | null
}

const App: React.FC = () => {
    const { isAuthenticated, isLoading: authLoading, token, logout, user } = useAuth()
    const [activeTab, setActiveTab] = useState<'chat' | 'documents'>('chat')
    const [messages, setMessages] = useState<ChatMessage[]>([])
    const [documents, setDocuments] = useState<Document[]>([])
    const [isLoading, setIsLoading] = useState(false)
    const [processingStatuses, setProcessingStatuses] = useState<Map<string, ProcessingStatus>>(new Map())
    const [isGenerating, setIsGenerating] = useState(false)
    const [currentGenerationId, setCurrentGenerationId] = useState<string | null>(null)
    const inputRef = useRef<HTMLInputElement>(null)
    const messagesContainerRef = useRef<HTMLDivElement>(null)
    const inputValueRef = useRef('')
    const fileInputRef = useRef<HTMLInputElement>(null)
    const uploadedFilesContainerRef = useRef<HTMLDivElement>(null)

    // New state for file upload system
    const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([])
    const [isProcessing, setIsProcessing] = useState(false)
    const [showSuccessMessage, setShowSuccessMessage] = useState(false)

    // Backend readiness state
    const [isBackendReady, setIsBackendReady] = useState(false)
    const [backendStatus, setBackendStatus] = useState('Kontroluji stav AI systému...')

    // Chat management state
    const [chats, setChats] = useState<Chat[]>([])
    const [currentChatId, setCurrentChatId] = useState<string | null>(null)
    const [isTemporaryMode, setIsTemporaryMode] = useState(false)

    // Scroll to bottom function - must be before early returns (Rules of Hooks)
    const scrollToBottom = useCallback(() => {
        if (messagesContainerRef.current) {
            messagesContainerRef.current.scrollTop = messagesContainerRef.current.scrollHeight
        }
    }, [])

    // Check backend readiness
    const checkBackendReadiness = async () => {
        const maxAttempts = Math.ceil(config.backendCheckTimeoutMs / config.backendCheckIntervalMs)
        let attempts = 0

        const checkHealth = async (): Promise<boolean> => {
            try {
                setBackendStatus('Inicializuji AI systém...')
                const response = await fetch(buildApiUrl(config.apiEndpoints.health), {
                    method: 'GET',
                    signal: AbortSignal.timeout(5000),
                })

                if (response.ok) {
                    const data = await response.json()
                    // Check both status and system_ready fields
                    if ((data.status === 'healthy' || data.status === 'ok') && data.system_ready === true) {
                        setBackendStatus('AI systém je připraven!')
                        setIsBackendReady(true)
                        return true
                    } else if (data.status === 'healthy' || data.status === 'ok') {
                        // Backend is running but system is not ready yet
                        setBackendStatus('AI systém se inicializuje...')
                        return false
                    }
                }
                return false
            } catch (error) {
                return false
            }
        }

        // Try to connect to backend
        while (attempts < maxAttempts) {
            const isHealthy = await checkHealth()
            if (isHealthy) {
                return
            }

            attempts++
            setBackendStatus(`AI systém se načítá... (${attempts}/${maxAttempts})`)
            await new Promise((resolve) => setTimeout(resolve, config.backendCheckIntervalMs))
        }

        // If still not ready after timeout, show error
        setBackendStatus('⚠️ AI systém se nepodařilo načíst po 30 pokusech. Zkontrolujte prosím, zda běží backend server a zda jsou správně nakonfigurovány služby (Ollama, Neo4j).')
        setIsBackendReady(false) // Don't allow usage if backend is not responding

        // Add an error message to the chat
        const errorMessage: ChatMessage = {
            id: `error-${Date.now()}`,
            content:
                '⚠️ **Chyba při načítání AI systému**\n\nAI systém se nepodařilo načíst po 30 pokusech (30 sekund).\n\n**Možné příčiny:**\n- Backend server neběží nebo je nedostupný\n- Ollama služba není spuštěná\n- Neo4j databáze není dostupná\n- Chybí některé závislosti\n\n**Co můžete udělat:**\n1. Zkontrolujte, zda běží backend server (port 8000)\n2. Ověřte, že běží Ollama: `ollama serve`\n3. Zkontrolujte Neo4j databázi: http://localhost:7474\n4. Zkuste restartovat aplikaci\n\nPokud problém přetrvává, podívejte se do logs: `server.log`',
            sender: 'assistant',
            timestamp: new Date().toISOString(),
        }
        setMessages((prev) => [...prev, errorMessage])
    }

    // No need to poll processing statuses since files are processed immediately

    const loadChats = async () => {
        try {
            const response = await fetchWithAuth(buildApiUrl(config.apiEndpoints.chats), { method: 'GET' }, token)
            if (response.ok) {
                const data = await response.json()
                const loadedChats: Chat[] = data.chats || []
                setChats(loadedChats)

                // If no current chat selected, select the most recent one
                if (!currentChatId && loadedChats.length > 0) {
                    setCurrentChatId(loadedChats[0].id)
                }
            }
        } catch (error) {
            console.error('Error loading chats:', error)
        }
    }

    const loadMessages = async (chatId: string | null) => {
        if (!chatId) {
            setMessages([])
            return
        }

        try {
            const response = await fetchWithAuth(buildApiUrl(config.apiEndpoints.chatMessages(chatId)), { method: 'GET' }, token)
            if (response.ok) {
                const msgs: ChatMessage[] = await response.json()
                setMessages(msgs)
                // Scroll to bottom after loading messages
                setTimeout(() => scrollToBottom(), 100)
            }
        } catch (error) {
            console.error('Error loading messages:', error)
        }
    }

    const createChat = async (isTemporary: boolean = false) => {
        try {
            const response = await fetchWithAuth(
                buildApiUrl(config.apiEndpoints.chats),
                {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ is_temporary: isTemporary }),
                },
                token
            )
            if (response.ok) {
                const newChat: Chat = await response.json()
                // Update chats list
                setChats((prev) => [newChat, ...prev])
                // Set as current chat
                setCurrentChatId(newChat.id)
                // Clear messages - they will be loaded when currentChatId changes
                setMessages([])
                return newChat
            }
        } catch (error) {
            console.error('Error creating chat:', error)
        }
        return null
    }

    const deleteChat = async (chatId: string) => {
        // Show confirmation dialog
        const chatToDelete = chats.find((c) => c.id === chatId)
        const chatName = chatToDelete?.name || 'tento chat'
        if (!window.confirm(`Opravdu chcete smazat chat "${chatName}"? Tato akce je nevratná.`)) {
            return
        }

        try {
            const response = await fetchWithAuth(buildApiUrl(config.apiEndpoints.chatDelete(chatId)), { method: 'DELETE' }, token)
            if (response.ok) {
                // Update chats list
                const updatedChats = chats.filter((c) => c.id !== chatId)
                setChats(updatedChats)

                // If deleted chat was current, switch to another or clear
                if (currentChatId === chatId) {
                    if (updatedChats.length > 0) {
                        setCurrentChatId(updatedChats[0].id)
                    } else {
                        setCurrentChatId(null)
                        setMessages([])
                    }
                }
            } else {
                const errorData = await response.json().catch(() => ({}))
                alert(`Chyba při mazání chatu: ${errorData.detail || 'Neznámá chyba'}`)
            }
        } catch (error) {
            console.error('Error deleting chat:', error)
            alert('Chyba při mazání chatu. Zkuste to prosím znovu.')
        }
    }

    const switchChat = async (chatId: string) => {
        setCurrentChatId(chatId)
        await loadMessages(chatId)
    }

    const updateChatName = async (chatId: string, newName: string) => {
        try {
            const response = await fetchWithAuth(
                buildApiUrl(config.apiEndpoints.chatUpdateName(chatId)),
                {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ name: newName }),
                },
                token
            )
            if (response.ok) {
                const updatedChat: Chat = await response.json()
                setChats((prev) => prev.map((c) => (c.id === chatId ? updatedChat : c)))
            }
        } catch (error) {
            console.error('Error updating chat name:', error)
        }
    }

    const loadDocuments = async () => {
        try {
            const response = await fetchWithAuth(buildApiUrl(config.apiEndpoints.documents), { method: 'GET' }, token)
            if (response.ok) {
                const docs: Document[] = await response.json()
                setDocuments(docs)
            }
        } catch (error) {}
    }

    // Load initial data - must be before early returns (Rules of Hooks)
    useEffect(() => {
        if (isAuthenticated && token) {
            checkBackendReadiness()
            loadChats()
            loadDocuments()
        }
    }, [isAuthenticated, token])

    // Load messages when chat changes
    useEffect(() => {
        if (currentChatId && isAuthenticated && token) {
            loadMessages(currentChatId)
        }
    }, [currentChatId, isAuthenticated, token])

    const sendMessage = useCallback(async () => {
        const messageContent = inputValueRef.current.trim()
        if (!messageContent || isLoading || isGenerating) return

        // Ensure we have a chat
        let chatId = currentChatId
        if (!chatId) {
            const newChat = await createChat(isTemporaryMode)
            if (!newChat) {
                return
            }
            chatId = newChat.id
            // Reload chats to ensure the new chat appears in the list
            await loadChats()
            // Ensure currentChatId is set and wait for state to update
            setCurrentChatId(chatId)
            // Small delay to ensure state is updated and useEffect fires
            await new Promise((resolve) => setTimeout(resolve, 150))
        }

        setIsLoading(true)
        setIsGenerating(true)

        // Generate IDs for both messages
        const userMessageId = `user-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
        const aiMessageId = `ai-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`

        // Set current generation ID for stop functionality
        setCurrentGenerationId(aiMessageId)

        // Create user message
        const userMessage: ChatMessage = {
            id: userMessageId,
            content: messageContent,
            sender: 'user',
            timestamp: new Date().toISOString(),
        }

        // Create loading AI message
        const loadingMessage: ChatMessage = {
            id: aiMessageId,
            content: '',
            sender: 'assistant',
            timestamp: new Date().toISOString(),
            loading: true,
        }

        // Add both messages in a single state update to prevent multiple re-renders
        setMessages((prev) => [...prev, userMessage, loadingMessage])

        // Clear input after adding messages to prevent focus loss
        if (inputRef.current) {
            inputRef.current.value = ''
            inputValueRef.current = ''
        }

        // Ensure input stays focused
        if (inputRef.current) {
            inputRef.current.focus()
        }

        // Scroll to bottom after adding messages
        setTimeout(() => scrollToBottom(), 100)

        try {
            // Call backend asynchronously
            const response = await fetchWithAuth(
                buildApiUrl(config.apiEndpoints.chatSendMessage(chatId)),
                {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ content: messageContent }),
                },
                token
            )

            if (response.ok) {
                // Reload chats to get updated names and ensure new chat is visible
                await loadChats()

                // Reload messages from server to ensure consistency
                await loadMessages(chatId)

                // Scroll to bottom after receiving response
                setTimeout(() => scrollToBottom(), 100)
            } else {
                throw new Error(`Server responded with ${response.status}: ${response.statusText}`)
            }
        } catch (error) {
            // Replace loading message with error message
            const errorContent =
                error instanceof Error
                    ? `⚠️ **Chyba při komunikaci s AI**\n\nNepodařilo se získat odpověď od AI systému.\n\n**Chyba:** ${error.message}\n\nZkuste:\n1. Znovu odeslat zprávu\n2. Zkontrolovat, zda běží backend\n3. Restartovat aplikaci`
                    : '⚠️ **Chyba při komunikaci s AI**\n\nNepodařilo se získat odpověď od AI systému. Zkuste to prosím znovu.'

            setMessages((prev) =>
                prev.map((msg) =>
                    msg.id === aiMessageId
                        ? {
                              ...msg,
                              content: errorContent,
                              loading: false,
                          }
                        : msg
                )
            )

            // Scroll to bottom after error
            setTimeout(() => scrollToBottom(), 100)
        } finally {
            setIsLoading(false)
            setIsGenerating(false)
            setCurrentGenerationId(null)
            // Refocus input after processing is complete
            if (inputRef.current) {
                inputRef.current.focus()
            }
        }
    }, [isLoading, isGenerating, token, scrollToBottom, currentChatId, isTemporaryMode, createChat, loadChats])

    const stopGeneration = useCallback(() => {
        if (isGenerating && currentGenerationId) {
            // Remove the loading message from chat
            setMessages((prev) => prev.filter((msg) => msg.id !== currentGenerationId))
            setIsGenerating(false)
            setCurrentGenerationId(null)
            setIsLoading(false)

            // Refocus input after stopping
            if (inputRef.current) {
                inputRef.current.focus()
            }
        }
    }, [isGenerating, currentGenerationId])

    // Show login if not authenticated
    if (authLoading) {
        return (
            <div className="min-h-screen bg-gray-100 flex items-center justify-center">
                <div className="text-center">
                    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
                    <p className="text-gray-600">Loading...</p>
                </div>
            </div>
        )
    }

    if (!isAuthenticated) {
        return <Login />
    }

    // Separate input component to prevent re-renders
    const ChatInputComponent = React.memo(({ isLoading, isGenerating, isBackendReady, backendStatus, sendMessage, stopGeneration }: { isLoading: boolean; isGenerating: boolean; isBackendReady: boolean; backendStatus: string; sendMessage: () => void; stopGeneration: () => void }) => {
        const isDisabled = isLoading || isGenerating || !isBackendReady

        return (
            <div className="border-t border-gray-200 p-4 bg-white">
                {!isBackendReady && (
                    <div className="mb-3 p-3 bg-yellow-50 border border-yellow-200 rounded-lg flex items-center space-x-2">
                        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-yellow-600"></div>
                        <span className="text-sm text-yellow-800">{backendStatus}</span>
                    </div>
                )}
                <div className="flex space-x-2">
                    <input
                        ref={inputRef}
                        type="text"
                        defaultValue=""
                        onChange={(e) => {
                            inputValueRef.current = e.target.value
                        }}
                        onKeyDown={(e) => {
                            if (e.key === 'Enter' && !isDisabled && inputValueRef.current.trim()) {
                                e.preventDefault()
                                sendMessage()
                            }
                        }}
                        placeholder={isBackendReady ? 'Zeptejte se na něco o svých dokumentech...' : 'Čekám na načtení AI...'}
                        className="flex-1 px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                        disabled={isDisabled}
                    />
                    {isGenerating ? (
                        <button onClick={stopGeneration} className="px-4 py-2 bg-red-500 text-white rounded-lg hover:bg-red-600 transition-colors flex items-center space-x-2">
                            <Square size={20} />
                            <span>Stop</span>
                        </button>
                    ) : (
                        <button
                            onClick={() => {
                                if (inputValueRef.current.trim()) {
                                    sendMessage()
                                }
                            }}
                            disabled={isDisabled}
                            className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                        >
                            <Send size={20} />
                        </button>
                    )}
                </div>
            </div>
        )
    })

    const handleFileInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        const files = event.target.files
        if (files && files.length > 0) {
            handleFileUpload(files)
        }
        // Reset the input value so the same file can be selected again
        if (fileInputRef.current) {
            fileInputRef.current.value = ''
        }
    }

    const handleFileUpload = async (files?: FileList) => {
        // Clear success message when new files are being uploaded
        setShowSuccessMessage(false)

        let filesToProcess: File[] = []

        if (files) {
            const allFiles = Array.from(files)
            const unsupportedFiles = allFiles.filter((file) => {
                const extension = file.name.split('.').pop()?.toLowerCase()
                const supportedExtensions = ['.pdf', '.docx', '.pptx', '.xlsx', '.html', '.htm', '.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif', '.wav', '.mp3', '.vtt', '.txt', '.md']
                return !supportedExtensions.includes(`.${extension}`)
            })

            if (unsupportedFiles.length > 0) {
                const unsupportedNames = unsupportedFiles.map((f) => f.name).join(', ')
                alert(`Nepodporované typy souborů: ${unsupportedNames}\n\nPodporované typy: PDF, DOCX, PPTX, XLSX, HTML, obrázky (PNG, JPG, TIFF, BMP, GIF), audio (WAV, MP3, VTT), TXT, MD`)
            }

            filesToProcess = allFiles.filter((file) => {
                const extension = file.name.split('.').pop()?.toLowerCase()
                const supportedExtensions = ['.pdf', '.docx', '.pptx', '.xlsx', '.html', '.htm', '.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif', '.wav', '.mp3', '.vtt', '.txt', '.md']
                const isSupported = supportedExtensions.includes(`.${extension}`)
                return isSupported
            })
        } else {
            // Use HTML file input for file selection
            if (fileInputRef.current) {
                fileInputRef.current.click()
            }
            return // The onchange handler will process the files
        }

        const newFiles: UploadedFile[] = filesToProcess.map((file) => ({
            file,
            id: Date.now().toString() + Math.random().toString(36).substr(2, 9),
            status: 'pending' as const,
            progress: 0,
            steps: [
                {
                    id: 'upload',
                    name: 'Nahrávání souboru',
                    description: 'Odesílání souboru na server...',
                    status: 'pending',
                    progress: 0,
                },
                {
                    id: 'extract',
                    name: 'Extrakce obsahu',
                    description: 'Extrakce textu, tabulek a obrázků...',
                    status: 'pending',
                    progress: 0,
                },
                {
                    id: 'chunk',
                    name: 'Rozdělení na části',
                    description: 'Rozdělení dokumentu na menší části...',
                    status: 'pending',
                    progress: 0,
                },
                {
                    id: 'embed',
                    name: 'Vytvoření vektorů',
                    description: 'Generování vektorových reprezentací...',
                    status: 'pending',
                    progress: 0,
                },
                {
                    id: 'index',
                    name: 'Indexování',
                    description: 'Uložení do databáze pro vyhledávání...',
                    status: 'pending',
                    progress: 0,
                },
            ],
        }))

        if (newFiles.length > 0) {
            setUploadedFiles((prev) => {
                return [...prev, ...newFiles]
            })
        }
    }

    const removeFile = (fileId: string) => {
        setUploadedFiles((prev) => prev.filter((f) => f.id !== fileId))
    }

    const processFiles = async () => {
        if (uploadedFiles.length === 0) return

        setIsProcessing(true)
        setShowSuccessMessage(false)

        // Process files sequentially - upload quickly, then start processing in background
        for (const uploadedFile of uploadedFiles) {
            try {
                // Update status to uploading (not processing yet)
                setUploadedFiles((prev) => prev.map((f) => (f.id === uploadedFile.id ? { ...f, status: 'uploading', progress: 0 } : f)))

                const formData = new FormData()
                formData.append('file', uploadedFile.file)

                const controller = new AbortController()
                // Longer timeout for upload to handle larger files
                const timeoutId = setTimeout(() => controller.abort(), config.uploadTimeoutMs)

                const response = await fetchWithAuth(
                    buildApiUrl(config.apiEndpoints.upload),
                    {
                        method: 'POST',
                        body: formData,
                        signal: controller.signal,
                    },
                    token
                )
                clearTimeout(timeoutId)

                if (response.ok) {
                    // Update status to processing and start polling
                    setUploadedFiles((prev) => prev.map((f) => (f.id === uploadedFile.id ? { ...f, status: 'processing', progress: 0 } : f)))
                    startStatusPolling(uploadedFile.id, uploadedFile.file.name)
                } else {
                    const errorText = await response.text()
                    await updateStep(uploadedFile.id, 'upload', 'error', 0, errorText)
                    setUploadedFiles((prev) => prev.map((f) => (f.id === uploadedFile.id ? { ...f, status: 'error', error: `Upload failed: ${errorText}` } : f)))
                }
            } catch (error) {
                let errorMessage = 'Unknown error'

                if (error instanceof Error) {
                    if (error.name === 'AbortError') {
                        errorMessage = 'Timeout při nahrávání - server příliš dlouho odpovídá. Zkuste to znovu.'
                    } else {
                        errorMessage = error.message
                    }
                } else {
                    errorMessage = String(error)
                }

                await updateStep(uploadedFile.id, 'upload', 'error', 0, errorMessage)
                setUploadedFiles((prev) => prev.map((f) => (f.id === uploadedFile.id ? { ...f, status: 'error', error: `Upload error: ${errorMessage}` } : f)))
            }
        }
    }

    const startStatusPolling = (fileId: string, filename: string) => {
        const maxPollingTime = config.maxPollingTimeMs
        const startTime = Date.now()
        let consecutiveFailures = 0
        const maxConsecutiveFailures = config.maxConsecutiveFailures

        const pollWithBackoff = async () => {
            // Check if we've been polling too long
            if (Date.now() - startTime > maxPollingTime) {
                setUploadedFiles((prev) =>
                    prev.map((f) => {
                        if (f.id === fileId) {
                            return {
                                ...f,
                                status: 'error',
                                error: 'Timeout při zpracování - dokument byl příliš velký nebo složitý. Zkuste to znovu nebo kontaktujte podporu.',
                            }
                        }
                        return f
                    })
                )
                setIsProcessing(false)
                clearInterval(pollInterval)
                return
            }
            try {
                const controller = new AbortController()
                const timeoutId = setTimeout(() => controller.abort(), 120000) // 2 minute timeout for status check (increased for large file processing)

                const response = await fetchWithAuth(
                    buildApiUrl(config.apiEndpoints.uploadStatus(filename)),
                    {
                        signal: controller.signal,
                    },
                    token
                )
                clearTimeout(timeoutId)

                // Reset failure counter on successful request
                consecutiveFailures = 0

                if (response.ok) {
                    const status = await response.json()

                    // Update file status based on backend status
                    setUploadedFiles((prev) =>
                        prev.map((f) => {
                            if (f.id === fileId) {
                                const updatedFile = { ...f }

                                // Update overall status
                                if (status.status === 'completed') {
                                    updatedFile.status = 'completed'
                                    updatedFile.progress = 100
                                    clearInterval(pollInterval)

                                    // Check if all files are completed after state update
                                    setTimeout(() => {
                                        setUploadedFiles((currentFiles) => {
                                            const allCompleted = currentFiles.every((f) => f.status === 'completed' || f.status === 'error')
                                            if (allCompleted) {
                                                setShowSuccessMessage(true)
                                                setIsProcessing(false)
                                                // Reload documents and clear files after delay
                                                setTimeout(() => {
                                                    loadDocuments()
                                                    setUploadedFiles([])
                                                }, config.successMessageDisplayMs)
                                            }
                                            return currentFiles
                                        })
                                    }, 1000)
                                } else if (status.status === 'error') {
                                    updatedFile.status = 'error'
                                    updatedFile.error = status.error || 'Zpracování selhalo - zkuste to znovu nebo kontaktujte podporu'
                                    clearInterval(pollInterval)

                                    // Check if all files are completed after error
                                    setTimeout(() => {
                                        setUploadedFiles((currentFiles) => {
                                            const allCompleted = currentFiles.every((f) => f.status === 'completed' || f.status === 'error')
                                            if (allCompleted) {
                                                setIsProcessing(false)
                                                // Reload documents - keep files visible so user can see errors
                                                loadDocuments()
                                            }
                                            return currentFiles
                                        })
                                    }, 1000)
                                } else {
                                    updatedFile.status = 'processing'
                                    updatedFile.progress = status.progress || 0
                                }

                                // Update steps based on backend status
                                if (status.steps) {
                                    updatedFile.steps = status.steps.map((step: any) => ({
                                        id: step.id,
                                        name: step.name,
                                        description: step.description || '',
                                        status: step.status,
                                        progress: step.progress || 0,
                                        error: step.error,
                                    }))
                                }

                                return updatedFile
                            }
                            return f
                        })
                    )
                } else {
                    // If status endpoint fails, increment failure counter but don't give up immediately
                    consecutiveFailures++

                    if (consecutiveFailures >= maxConsecutiveFailures) {
                        setUploadedFiles((prev) =>
                            prev.map((f) => {
                                if (f.id === fileId) {
                                    return {
                                        ...f,
                                        status: 'error',
                                        error: `Kontrola stavu selhala po ${maxConsecutiveFailures} pokusech. Server neodpovídá správně.`,
                                    }
                                }
                                return f
                            })
                        )
                        setIsProcessing(false)
                        clearInterval(pollInterval)
                    } else {
                        // Continue polling but show a warning
                        setUploadedFiles((prev) =>
                            prev.map((f) => {
                                if (f.id === fileId) {
                                    return {
                                        ...f,
                                        status: 'processing',
                                        error: `Kontrola stavu selhala (pokus ${consecutiveFailures}/${maxConsecutiveFailures}), zkouším znovu...`,
                                    }
                                }
                                return f
                            })
                        )
                    }
                }
            } catch (error) {
                // Increment failure counter on network errors too
                consecutiveFailures++

                if (consecutiveFailures >= maxConsecutiveFailures) {
                    let errorMessage = 'Neznámá chyba'

                    if (error instanceof Error) {
                        if (error.name === 'AbortError') {
                            errorMessage = 'Timeout požadavku - server příliš dlouho odpovídá'
                        } else if (error.message.includes('Load failed')) {
                            errorMessage = 'Připojení selhalo - zkontrolujte, zda server běží'
                        } else if (error.message.includes('Failed to fetch')) {
                            errorMessage = 'Chyba sítě - zkontrolujte své připojení'
                        } else {
                            errorMessage = `Chyba sítě: ${error.message}`
                        }
                    } else {
                        errorMessage = `Chyba sítě: ${String(error)}`
                    }

                    // Update file status to show network error
                    setUploadedFiles((prev) =>
                        prev.map((f) => {
                            if (f.id === fileId) {
                                return {
                                    ...f,
                                    status: 'error',
                                    error: errorMessage,
                                }
                            }
                            return f
                        })
                    )
                    setIsProcessing(false)
                    clearInterval(pollInterval)
                } else {
                    // Continue polling but show a warning
                    setUploadedFiles((prev) =>
                        prev.map((f) => {
                            if (f.id === fileId) {
                                return {
                                    ...f,
                                    status: 'processing',
                                    error: `Chyba sítě (pokus ${consecutiveFailures}/${maxConsecutiveFailures}), zkouším znovu...`,
                                }
                            }
                            return f
                        })
                    )
                }
            }
        }

        // Start polling with reasonable interval
        const pollInterval = setInterval(async () => {
            await pollWithBackoff()
        }, config.statusPollIntervalMs)
    }

    const updateStep = async (fileId: string, stepId: string, status: 'pending' | 'in_progress' | 'completed' | 'error', progress: number, error?: string) => {
        setUploadedFiles((prev) =>
            prev.map((f) =>
                f.id === fileId
                    ? {
                          ...f,
                          steps: f.steps.map((step) => (step.id === stepId ? { ...step, status, progress, error } : step)),
                      }
                    : f
            )
        )
    }

    const clearFiles = () => {
        setUploadedFiles([])
        setShowSuccessMessage(false)
    }

    const deleteDocument = async (documentId: string) => {
        try {
            const response = await fetchWithAuth(buildApiUrl(config.apiEndpoints.deleteDocument(documentId)), { method: 'DELETE' }, token)
            if (response.ok) {
                await loadDocuments()
                setProcessingStatuses((prev) => {
                    const newMap = new Map(prev)
                    newMap.delete(documentId)
                    return newMap
                })
            }
        } catch (error) {}
    }

    const formatFileSize = (bytes: number): string => {
        const sizes = ['Bytes', 'KB', 'MB', 'GB']
        if (bytes === 0) return '0 Bytes'
        const i = Math.floor(Math.log(bytes) / Math.log(1024))
        return Math.round((bytes / Math.pow(1024, i)) * 100) / 100 + ' ' + sizes[i]
    }

    const formatTimestamp = (timestamp: string): string => {
        return new Date(timestamp).toLocaleTimeString()
    }

    // Parse table data from response text
    const parseTableData = (content: string): ParsedMessage => {
        // Updated regex to capture filename if present
        const tableRegex = /Page (\d+), Type: (\w+), Chunk: (\w+)(?:, Soubor: ([^)]+))?/g
        const tables: TableData[] = []
        let match

        while ((match = tableRegex.exec(content)) !== null) {
            tables.push({
                page: match[1],
                type: match[2],
                chunk: match[3],
                filename: match[4] || undefined,
            })
        }

        // Remove table references from text content
        const textContent = content.replace(tableRegex, '').trim()

        return {
            text: textContent,
            tables,
        }
    }

    // Table display component
    const TableDisplay: React.FC<{ tables: TableData[] }> = ({ tables }) => {
        if (tables.length === 0) return null

        return (
            <div className="mt-3 space-y-2">
                <div className="text-xs font-medium text-gray-600 mb-2">📊 Table References:</div>
                {tables.map((table, index) => (
                    <div key={index} className="bg-blue-50 border border-blue-200 rounded-lg p-3">
                        <div className="flex items-center space-x-2 text-sm">
                            <span className="font-medium text-blue-800">Page {table.page}</span>
                            <span className="text-blue-600">•</span>
                            <span className="text-blue-700">{table.type}</span>
                            <span className="text-blue-600">•</span>
                            <span className="text-blue-700">Chunk: {table.chunk}</span>
                            {table.filename && (
                                <>
                                    <span className="text-blue-600">•</span>
                                    <span className="text-blue-700 font-medium">📄 {table.filename}</span>
                                </>
                            )}
                        </div>
                    </div>
                ))}
            </div>
        )
    }

    // Formatted message component for rendering markdown/HTML
    const FormattedMessage: React.FC<{ content: string; isUser: boolean }> = ({ content, isUser }) => {
        // For user messages, just show plain text
        if (isUser) {
            return <p className="text-sm whitespace-pre-line">{content}</p>
        }

        // For assistant messages, render markdown and HTML
        return (
            <div className="text-sm">
                <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    rehypePlugins={[rehypeRaw]}
                    components={{
                        // Style headings
                        h1: ({ node, ...props }) => <h1 className="text-lg font-bold mb-2 mt-3 first:mt-0" {...props} />,
                        h2: ({ node, ...props }) => <h2 className="text-base font-bold mb-2 mt-3 first:mt-0" {...props} />,
                        h3: ({ node, ...props }) => <h3 className="text-sm font-bold mb-1 mt-2 first:mt-0" {...props} />,
                        // Style paragraphs
                        p: ({ node, ...props }) => <p className="mb-2 last:mb-0 whitespace-pre-wrap" {...props} />,
                        // Style lists
                        ul: ({ node, ...props }) => <ul className="list-disc list-inside mb-2 space-y-1" {...props} />,
                        ol: ({ node, ...props }) => <ol className="list-decimal list-inside mb-2 space-y-1" {...props} />,
                        li: ({ node, ...props }) => <li className="ml-2" {...props} />,
                        // Style bold and italic
                        strong: ({ node, ...props }) => <strong className="font-bold" {...props} />,
                        em: ({ node, ...props }) => <em className="italic" {...props} />,
                        // Style code blocks
                        code: ({ node, inline, ...props }: any) => (inline ? <code className="bg-gray-300 px-1 py-0.5 rounded text-xs font-mono" {...props} /> : <code className="block bg-gray-300 p-2 rounded text-xs font-mono overflow-x-auto" {...props} />),
                        pre: ({ node, ...props }) => <pre className="bg-gray-300 p-2 rounded text-xs font-mono overflow-x-auto mb-2" {...props} />,
                        // Style blockquotes
                        blockquote: ({ node, ...props }) => <blockquote className="border-l-4 border-gray-400 pl-2 italic my-2" {...props} />,
                        // Style links
                        a: ({ node, ...props }) => <a className="text-blue-600 underline" target="_blank" rel="noopener noreferrer" {...props} />,
                        // Style divs (for HTML content)
                        div: ({ node, ...props }: any) => <div className="mb-2 last:mb-0" {...props} />,
                        // Style tables
                        table: ({ node, ...props }: any) => (
                            <div className="overflow-x-auto my-3">
                                <table className="min-w-full border-collapse border border-gray-300 text-xs" {...props} />
                            </div>
                        ),
                        thead: ({ node, ...props }: any) => <thead className="bg-gray-100" {...props} />,
                        tbody: ({ node, ...props }: any) => <tbody {...props} />,
                        tr: ({ node, ...props }: any) => <tr className="border-b border-gray-200 hover:bg-gray-50" {...props} />,
                        th: ({ node, ...props }: any) => <th className="border border-gray-300 px-2 py-1 text-left font-semibold" {...props} />,
                        td: ({ node, ...props }: any) => <td className="border border-gray-300 px-2 py-1" {...props} />,
                    }}
                >
                    {content}
                </ReactMarkdown>
            </div>
        )
    }

    // Step progress component
    const StepProgress: React.FC<{ steps: UploadStep[] }> = ({ steps }) => {
        return (
            <div className="mt-3 space-y-2">
                <div className="text-xs font-medium text-gray-600 mb-2">📋 Postup zpracování:</div>
                {steps.map((step) => (
                    <div
                        key={step.id}
                        className={`flex items-center space-x-3 p-2 rounded-lg transition-colors ${
                            step.status === 'in_progress' ? 'bg-blue-50 border border-blue-200' : step.status === 'completed' ? 'bg-green-50 border border-green-200' : step.status === 'error' ? 'bg-red-50 border border-red-200' : 'bg-gray-50 border border-gray-200'
                        }`}
                    >
                        <div className="flex-shrink-0">
                            {step.status === 'completed' && <CheckCircle size={16} className="text-green-500" />}
                            {step.status === 'in_progress' && <Clock size={16} className="text-blue-500 animate-spin" />}
                            {step.status === 'error' && <AlertCircle size={16} className="text-red-500" />}
                            {step.status === 'pending' && <div className="w-4 h-4 rounded-full border-2 border-gray-300" />}
                        </div>
                        <div className="flex-1 min-w-0">
                            <div className="flex items-center justify-between">
                                <span className={`text-sm font-medium ${step.status === 'completed' ? 'text-green-700' : step.status === 'in_progress' ? 'text-blue-700' : step.status === 'error' ? 'text-red-700' : 'text-gray-600'}`}>{step.name}</span>
                                {(step.status === 'in_progress' || step.status === 'completed') && <span className="text-xs text-gray-500 ml-2">{Math.round(step.progress * 100)}%</span>}
                            </div>
                            {step.description && <p className="text-xs text-gray-500 mt-1">{step.description}</p>}
                            {step.status === 'in_progress' && (
                                <div className="mt-1 w-full bg-gray-200 rounded-full h-1.5">
                                    <div className="bg-blue-500 h-1.5 rounded-full transition-all duration-300" style={{ width: `${Math.min(100, Math.max(0, step.progress * 100))}%` }} />
                                </div>
                            )}
                            {step.status === 'error' && step.error && <p className="text-xs text-red-600 mt-1">Chyba: {step.error}</p>}
                        </div>
                    </div>
                ))}
            </div>
        )
    }

    const ChatTab = () => {
        const [isEditingName, setIsEditingName] = useState(false)
        const [editedName, setEditedName] = useState('')
        const nameInputRef = useRef<HTMLInputElement>(null)
        const currentChat = chats.find((c) => c.id === currentChatId)

        const handleEditName = () => {
            if (currentChat) {
                setEditedName(currentChat.name)
                setIsEditingName(true)
                setTimeout(() => nameInputRef.current?.focus(), 0)
            }
        }

        const handleSaveName = async () => {
            if (currentChatId && editedName.trim()) {
                await updateChatName(currentChatId, editedName.trim())
                setIsEditingName(false)
            }
        }

        const handleCancelEdit = () => {
            setIsEditingName(false)
            setEditedName('')
        }

        return (
            <div className="flex flex-col h-full">
                {/* Header */}
                <div className="border-b border-gray-200 p-4 bg-white">
                    <div className="flex justify-between items-center">
                        <div className="flex-1 min-w-0">
                            {isEditingName ? (
                                <div className="flex items-center space-x-2">
                                    <input
                                        ref={nameInputRef}
                                        type="text"
                                        value={editedName}
                                        onChange={(e) => setEditedName(e.target.value)}
                                        onKeyDown={(e) => {
                                            if (e.key === 'Enter') handleSaveName()
                                            if (e.key === 'Escape') handleCancelEdit()
                                        }}
                                        className="flex-1 px-2 py-1 border border-blue-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
                                        maxLength={100}
                                    />
                                    <button onClick={handleSaveName} className="p-1 text-green-600 hover:bg-green-50 rounded" title="Uložit">
                                        <CheckCircle size={18} />
                                    </button>
                                    <button onClick={handleCancelEdit} className="p-1 text-gray-600 hover:bg-gray-50 rounded" title="Zrušit">
                                        <X size={18} />
                                    </button>
                                </div>
                            ) : (
                                <div className="flex items-center space-x-2">
                                    <h2 className="text-xl font-semibold text-gray-800 truncate">{currentChat?.name || 'Nový Chat'}</h2>
                                    {currentChat && (
                                        <>
                                            {currentChat.is_temporary && <span className="text-xs text-gray-500 bg-gray-200 px-2 py-0.5 rounded">Dočasný</span>}
                                            <button onClick={handleEditName} className="p-1 text-gray-400 hover:text-gray-600 rounded transition-colors" title="Upravit název">
                                                <Edit2 size={16} />
                                            </button>
                                        </>
                                    )}
                                </div>
                            )}
                        </div>
                    </div>
                </div>

                {/* Messages */}
                <div ref={messagesContainerRef} className="flex-1 overflow-y-auto p-4 space-y-4 custom-scrollbar">
                    {!isBackendReady ? (
                        <div className="text-center text-gray-500 mt-20">
                            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
                            <p className="text-lg font-medium">{backendStatus}</p>
                            <p className="text-sm mt-2">Počkejte prosím, zatímco se AI systém inicializuje...</p>
                        </div>
                    ) : messages.length === 0 ? (
                        <div className="text-center text-gray-500 mt-20">
                            <MessageCircle size={48} className="mx-auto mb-4 opacity-50" />
                            <p>Zatím žádné zprávy. Začněte konverzaci!</p>
                            <p className="text-sm mt-2">Nejprve nahrajte dokumenty (PDF, DOCX, PPTX, XLSX, HTML, obrázky, audio, TXT, MD), abyste se na ně mohli ptát.</p>
                        </div>
                    ) : (
                        messages.map((message) => {
                            const parsedMessage = parseTableData(message.content)
                            return (
                                <div key={message.id} className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}>
                                    <div className={`max-w-xs lg:max-w-md px-4 py-2 rounded-lg ${message.sender === 'user' ? 'bg-blue-500 text-white' : 'bg-gray-200 text-gray-800'}`}>
                                        {parsedMessage.text && <FormattedMessage content={parsedMessage.text} isUser={message.sender === 'user'} />}
                                        {message.sender === 'assistant' && parsedMessage.tables.length > 0 && <TableDisplay tables={parsedMessage.tables} />}
                                        {message.loading && (
                                            <div className="flex items-center mt-2">
                                                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-gray-400"></div>
                                                <span className="text-xs ml-2 opacity-75">Přemýšlím...</span>
                                            </div>
                                        )}
                                        <p className={`text-xs mt-1 opacity-75`}>{formatTimestamp(message.timestamp)}</p>
                                    </div>
                                </div>
                            )
                        })
                    )}
                </div>

                {/* Input */}
                <ChatInputComponent isLoading={isLoading} isGenerating={isGenerating} isBackendReady={isBackendReady} backendStatus={backendStatus} sendMessage={sendMessage} stopGeneration={stopGeneration} />
            </div>
        )
    }

    const DocumentsTab = () => (
        <div className="flex flex-col h-full overflow-hidden">
            {/* Header */}
            <div className="border-b border-gray-200 p-4 bg-white flex-shrink-0">
                <h2 className="text-xl font-semibold text-gray-800">Správa Dokumentů</h2>
                <p className="text-sm text-gray-600 mt-1">Nahrajte dokumenty (PDF, DOCX, PPTX, XLSX, HTML, obrázky, audio, TXT, MD) pro chat s nimi</p>
            </div>

            {/* Upload Area */}
            <div className="p-4 bg-gray-50 flex-shrink-0 overflow-y-auto max-h-[50vh]">
                <div
                    className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center upload-area cursor-pointer hover:border-blue-400 hover:bg-blue-50 transition-colors"
                    onDragOver={(e) => {
                        e.preventDefault()
                        e.currentTarget.classList.add('border-blue-400', 'bg-blue-50')
                    }}
                    onDragLeave={(e) => {
                        e.preventDefault()
                        e.currentTarget.classList.remove('border-blue-400', 'bg-blue-50')
                    }}
                    onDrop={(e) => {
                        e.preventDefault()
                        e.currentTarget.classList.remove('border-blue-400', 'bg-blue-50')
                        if (e.dataTransfer.files) {
                            handleFileUpload(e.dataTransfer.files)
                        }
                    }}
                    onClick={async () => {
                        await handleFileUpload()
                    }}
                >
                    <Upload size={48} className="mx-auto mb-4 text-gray-400" />
                    <p className="text-lg font-medium text-gray-700">Přetáhněte soubory sem nebo klikněte pro nahrání</p>
                    <p className="text-sm text-gray-500 mt-2">Podporuje PDF, DOCX, PPTX, XLSX, HTML, obrázky, audio, TXT a MD soubory do 10MB</p>
                </div>

                {/* Alternative Upload Button */}
                <div className="mt-4 text-center">
                    <button onClick={async () => await handleFileUpload()} className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors flex items-center space-x-2 mx-auto">
                        <Upload size={20} />
                        <span>Prohlédnout</span>
                    </button>
                </div>

                {/* Success Message */}
                {showSuccessMessage && (
                    <div className="mt-4 p-4 bg-green-50 border border-green-200 rounded-lg">
                        <div className="flex items-center space-x-2">
                            <CheckCircle size={20} className="text-green-600" />
                            <span className="text-green-800 font-medium">Všechny soubory byly úspěšně zpracovány!</span>
                        </div>
                        <p className="text-green-700 text-sm mt-1">Soubory byly nahrány a jsou připraveny pro chat.</p>
                    </div>
                )}

                {/* Uploaded Files Display */}
                {uploadedFiles.length > 0 && (
                    <div className="mt-4">
                        <div className="flex items-center justify-between mb-3">
                            <h3 className="text-sm font-medium text-gray-700">Vybrané Soubory ({uploadedFiles.length})</h3>
                            <div className="flex space-x-2">
                                {uploadedFiles.some((f) => f.status === 'pending') && (
                                    <button onClick={processFiles} disabled={isProcessing} className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-blue-400 disabled:cursor-not-allowed transition-colors flex items-center space-x-2">
                                        {isProcessing ? <Clock size={16} className="animate-spin" /> : <Play size={16} />}
                                        <span>{isProcessing ? 'Zpracovávám...' : 'Zpracovat Soubory'}</span>
                                    </button>
                                )}
                                <button
                                    onClick={clearFiles}
                                    disabled={isProcessing && uploadedFiles.some((f) => f.status === 'processing' || f.status === 'uploading')}
                                    className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors flex items-center space-x-2"
                                >
                                    <X size={16} />
                                    <span>Vymazat Vše</span>
                                </button>
                            </div>
                        </div>

                        {/* File Cards */}
                        <div ref={uploadedFilesContainerRef} className="space-y-3">
                            {uploadedFiles.map((uploadedFile) => (
                                <div key={uploadedFile.id} className="border border-gray-200 rounded-lg p-4 bg-white shadow-sm">
                                    <div className="flex items-start justify-between">
                                        <div className="flex-1">
                                            <div className="flex items-center space-x-2">
                                                <FileText size={20} className="text-blue-500" />
                                                <h4 className="font-medium text-gray-800 truncate">{uploadedFile.file.name}</h4>
                                                {uploadedFile.status === 'completed' && <CheckCircle size={16} className="text-green-500" />}
                                                {uploadedFile.status === 'error' && <AlertCircle size={16} className="text-red-500" />}
                                                {uploadedFile.status === 'processing' && <Clock size={16} className="text-yellow-500 animate-spin" />}
                                                {uploadedFile.status === 'uploading' && <Clock size={16} className="text-blue-500 animate-spin" />}
                                            </div>
                                            <p className="text-sm text-gray-500 mt-1">
                                                {formatFileSize(uploadedFile.file.size)} • {uploadedFile.file.type}
                                            </p>

                                            {/* Status Messages */}
                                            {uploadedFile.status === 'pending' && <div className="mt-2 p-2 bg-gray-50 border border-gray-200 rounded text-xs text-gray-700">⏳ Čeká na zpracování...</div>}

                                            {uploadedFile.status === 'uploading' && <div className="mt-2 p-2 bg-blue-50 border border-blue-200 rounded text-xs text-blue-700">📤 Nahrávání na server...</div>}

                                            {uploadedFile.status === 'processing' && (
                                                <>
                                                    <div className="mt-2 p-2 bg-yellow-50 border border-yellow-200 rounded text-xs text-yellow-700">⚙️ Zpracování dokumentu... {uploadedFile.progress > 0 && `(${Math.round(uploadedFile.progress * 100)}%)`}</div>
                                                    {uploadedFile.steps && uploadedFile.steps.length > 0 && <StepProgress steps={uploadedFile.steps} />}
                                                </>
                                            )}

                                            {uploadedFile.status === 'error' && uploadedFile.error && (
                                                <>
                                                    <div className="mt-2 p-2 bg-red-50 border border-red-200 rounded text-xs text-red-700">Chyba: {uploadedFile.error}</div>
                                                    {uploadedFile.steps && uploadedFile.steps.length > 0 && <StepProgress steps={uploadedFile.steps} />}
                                                </>
                                            )}

                                            {uploadedFile.status === 'processing' && uploadedFile.error && uploadedFile.error.includes('retrying') && <div className="mt-2 p-2 bg-yellow-50 border border-yellow-200 rounded text-xs text-yellow-700">⚠️ {uploadedFile.error}</div>}

                                            {uploadedFile.status === 'completed' && (
                                                <>
                                                    <div className="mt-2 p-2 bg-green-50 border border-green-200 rounded text-xs text-green-700">✅ Úspěšně zpracováno</div>
                                                    {uploadedFile.steps && uploadedFile.steps.length > 0 && <StepProgress steps={uploadedFile.steps} />}
                                                </>
                                            )}
                                        </div>

                                        {(uploadedFile.status === 'pending' || uploadedFile.status === 'error') && (
                                            <button onClick={() => removeFile(uploadedFile.id)} className="ml-4 p-2 text-red-500 hover:bg-red-50 rounded transition-colors" title="Odstranit soubor">
                                                <X size={16} />
                                            </button>
                                        )}
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                )}
            </div>

            {/* Documents List */}
            <div className="flex-1 overflow-y-auto p-4 custom-scrollbar">
                <h3 className="text-lg font-medium text-gray-800 mb-4">Zpracované Dokumenty</h3>
                {documents.length === 0 ? (
                    <div className="text-center text-gray-500 mt-20">
                        <FileText size={48} className="mx-auto mb-4 opacity-50" />
                        <p>Zatím žádné zpracované dokumenty</p>
                        <p className="text-sm mt-2">Nahrajte a zpracujte dokumenty (PDF, DOCX, PPTX, XLSX, HTML, obrázky, audio, TXT, MD) pro chat s nimi</p>
                    </div>
                ) : (
                    <div className="space-y-3">
                        {documents.map((doc) => {
                            const status = processingStatuses.get(doc.id)
                            return (
                                <div key={doc.id} className="border border-gray-200 rounded-lg p-4 bg-white shadow-sm">
                                    <div className="flex items-start justify-between">
                                        <div className="flex-1">
                                            <div className="flex items-center space-x-2">
                                                <FileText size={20} className="text-blue-500" />
                                                <h3 className="font-medium text-gray-800">{doc.name}</h3>
                                                {doc.processed ? <CheckCircle size={16} className="text-green-500" /> : <Clock size={16} className="text-yellow-500" />}
                                            </div>
                                            <p className="text-sm text-gray-500 mt-1">
                                                {formatFileSize(doc.size)} • Uploaded {new Date(doc.upload_time).toLocaleDateString()}
                                            </p>

                                            {/* Processing Status */}
                                            {status && !doc.processed && (
                                                <div className="mt-3">
                                                    <div className="flex items-center justify-between mb-1">
                                                        <span className="text-xs text-gray-600">{status.message}</span>
                                                        <span className="text-xs text-gray-600">{Math.round(status.progress * 100)}%</span>
                                                    </div>
                                                    <div className="w-full bg-gray-200 rounded-full h-2">
                                                        <div className="bg-blue-500 h-2 rounded-full transition-all duration-300" style={{ width: `${status.progress * 100}%` }}></div>
                                                    </div>
                                                </div>
                                            )}
                                        </div>

                                        <button onClick={() => deleteDocument(doc.id)} className="ml-4 p-2 text-red-500 hover:bg-red-50 rounded transition-colors">
                                            <Trash2 size={16} />
                                        </button>
                                    </div>
                                </div>
                            )
                        })}
                    </div>
                )}
            </div>
        </div>
    )

    return (
        <div className="flex h-screen bg-gray-100">
            {/* Hidden file input */}
            <input ref={fileInputRef} type="file" multiple accept=".pdf,.docx,.pptx,.xlsx,.html,.htm,.png,.jpg,.jpeg,.tiff,.bmp,.gif,.wav,.mp3,.vtt,.txt,.md" onChange={handleFileInputChange} style={{ display: 'none' }} />
            {/* Sidebar */}
            <div className="w-64 bg-slate-800 text-white flex flex-col">
                <div className="p-4 border-b border-slate-700">
                    <h1 className="text-xl font-bold">{config.appTitle}</h1>
                    <p className="text-sm text-slate-300">{config.appSubtitle}</p>
                </div>

                <nav className="flex-1 p-4">
                    <div className="space-y-2">
                        <button onClick={() => setActiveTab('chat')} className={`w-full flex items-center space-x-3 px-3 py-2 rounded-lg transition-colors ${activeTab === 'chat' ? 'bg-blue-600' : 'hover:bg-slate-700'}`}>
                            <MessageCircle size={20} />
                            <span>Chat</span>
                        </button>

                        <button onClick={() => setActiveTab('documents')} className={`w-full flex items-center space-x-3 px-3 py-2 rounded-lg transition-colors ${activeTab === 'documents' ? 'bg-blue-600' : 'hover:bg-slate-700'}`}>
                            <Upload size={20} />
                            <span>Dokumenty</span>
                            {documents.length > 0 && <span className="ml-auto bg-slate-600 text-xs px-2 py-1 rounded-full">{documents.length}</span>}
                        </button>
                    </div>
                </nav>

                <div className="p-4 border-t border-slate-700 space-y-2">
                    {user && (
                        <div className="px-3 py-2 text-sm text-slate-300">
                            Přihlášen jako: <span className="font-medium text-white">{user?.username || ''}</span>
                        </div>
                    )}
                    <button onClick={logout} className="w-full flex items-center space-x-3 px-3 py-2 rounded-lg hover:bg-red-600 transition-colors text-red-200 hover:text-white">
                        <LogOut size={20} />
                        <span>Odhlásit se</span>
                    </button>
                </div>
            </div>

            {/* Chat Sidebar - only show on chat tab */}
            {activeTab === 'chat' && (
                <div className="w-64 bg-gray-50 border-r border-gray-200 flex flex-col">
                    <div className="p-4 border-b border-gray-200 bg-white">
                        <div className="flex items-center justify-between mb-3">
                            <h2 className="text-lg font-semibold text-gray-800">Chaty</h2>
                            <div className="flex space-x-1">
                                <button onClick={() => createChat(false)} className="p-2 text-gray-600 hover:bg-gray-100 rounded transition-colors" title="Nový chat">
                                    <Plus size={18} />
                                </button>
                            </div>
                        </div>
                        <div className="flex items-center space-x-2">
                            <label className="flex items-center space-x-2 text-sm text-gray-600 cursor-pointer">
                                <input type="checkbox" checked={isTemporaryMode} onChange={(e) => setIsTemporaryMode(e.target.checked)} className="rounded" />
                                <span>Dočasný chat</span>
                            </label>
                        </div>
                    </div>

                    <div className="flex-1 overflow-y-auto p-2 space-y-1">
                        {chats.length === 0 ? (
                            <div className="text-center text-gray-500 mt-8 p-4">
                                <MessageCircle size={32} className="mx-auto mb-2 opacity-50" />
                                <p className="text-sm">Žádné chaty</p>
                                <p className="text-xs mt-1">Vytvořte nový chat</p>
                            </div>
                        ) : (
                            chats.map((chat) => (
                                <div key={chat.id} onClick={() => switchChat(chat.id)} className={`group p-3 rounded-lg cursor-pointer transition-colors ${currentChatId === chat.id ? 'bg-blue-100 border border-blue-300' : 'bg-white border border-gray-200 hover:bg-gray-50'}`}>
                                    <div className="flex items-start justify-between">
                                        <div className="flex-1 min-w-0">
                                            <div className="flex items-center space-x-2">
                                                <p className={`text-sm font-medium truncate ${currentChatId === chat.id ? 'text-blue-800' : 'text-gray-800'}`}>{chat.name}</p>
                                                {chat.is_temporary && <span className="text-xs text-gray-500 bg-gray-200 px-1.5 py-0.5 rounded">Dočasný</span>}
                                            </div>
                                            {chat.last_message_at && (
                                                <p className="text-xs text-gray-500 mt-1">
                                                    {new Date(chat.last_message_at).toLocaleDateString('cs-CZ', {
                                                        day: '2-digit',
                                                        month: '2-digit',
                                                        hour: '2-digit',
                                                        minute: '2-digit',
                                                    })}
                                                </p>
                                            )}
                                        </div>
                                        <button
                                            onClick={(e) => {
                                                e.stopPropagation()
                                                deleteChat(chat.id)
                                            }}
                                            className="ml-2 p-1.5 text-gray-400 hover:text-red-500 hover:bg-red-50 rounded transition-colors opacity-0 group-hover:opacity-100 flex-shrink-0"
                                            title="Smazat chat"
                                        >
                                            <Trash2 size={14} />
                                        </button>
                                    </div>
                                </div>
                            ))
                        )}
                    </div>
                </div>
            )}

            {/* Main Content */}
            <div className="flex-1 flex flex-col">{activeTab === 'chat' ? <ChatTab /> : <DocumentsTab />}</div>
        </div>
    )
}

export default App
