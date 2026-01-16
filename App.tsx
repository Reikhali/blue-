
import React, { useState, useRef, useCallback, useEffect } from 'react';
import { GoogleGenAI, Modality, LiveServerMessage } from '@google/genai';
import { createClient } from '@deepgram/sdk';
import { ConnectionStatus, Message } from './types';
import { encode, decode, decodeAudioData, blobToBase64, createAudioBlob } from './utils';

const GEMINI_MODEL = 'gemini-2.5-flash-native-audio-preview-12-2025';
const JPEG_QUALITY = 0.5; // Qualidade otimizada para análise
const FRAME_RATE = 1; // 1 frame por segundo para análise

const App: React.FC = () => {
  const [status, setStatus] = useState<ConnectionStatus>(ConnectionStatus.DISCONNECTED);
  const [messages, setMessages] = useState<Message[]>([]);
  const [isCapturing, setIsCapturing] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [mode, setMode] = useState<'screen' | 'camera'>('screen');
  const [userTranscript, setUserTranscript] = useState('');
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const sessionRef = useRef<any>(null);
  const sessionPromiseRef = useRef<Promise<any> | null>(null);
  const deepgramConnectionRef = useRef<any>(null);
  
  const inputAudioContextRef = useRef<AudioContext | null>(null);
  const outputAudioContextRef = useRef<AudioContext | null>(null);
  const audioOutputNodeRef = useRef<GainNode | null>(null);
  const nextStartTimeRef = useRef<number>(0);
  const sourcesRef = useRef<Set<AudioBufferSourceNode>>(new Set());
  const micStreamRef = useRef<MediaStream | null>(null);
  const captureIntervalRef = useRef<number | null>(null);

  useEffect(() => {
    if (window.location.protocol !== 'https:' && window.location.hostname !== 'localhost') {
      setErrorMsg("ERRO DE SEGURANÇA: O compartilhamento de tela requer conexão HTTPS.");
    }
  }, []);

  const addMessage = (role: 'user' | 'assistant', text: string) => {
    if (!text) return;
    setMessages(prev => {
        const last = prev[prev.length - 1];
        if (last && last.role === role && role === 'assistant') {
            const newPrev = [...prev];
            newPrev[newPrev.length - 1] = { ...last, text: last.text + text };
            return newPrev.slice(-5);
        }
        return [...prev, { role, text, timestamp: Date.now() }].slice(-5);
    });
  };

  const stopCapture = useCallback(() => {
    if (captureIntervalRef.current) clearInterval(captureIntervalRef.current);
    
    micStreamRef.current?.getTracks().forEach(t => t.stop());
    const videoStream = videoRef.current?.srcObject as MediaStream;
    videoStream?.getTracks().forEach(t => t.stop());
    videoRef.current!.srcObject = null;
    
    if (sessionRef.current) {
        try { sessionRef.current.close(); } catch(e){}
        sessionRef.current = null;
        sessionPromiseRef.current = null;
    }

    if (deepgramConnectionRef.current) {
        deepgramConnectionRef.current.finish();
        deepgramConnectionRef.current = null;
    }
    
    setIsCapturing(false);
    setIsSpeaking(false);
    setStatus(ConnectionStatus.DISCONNECTED);
    setUserTranscript('');
  }, []);
  
  const startAnalysis = async () => {
    try {
      setErrorMsg(null);
      if (!process.env.DEEPGRAM_API_KEY) {
        throw new Error("A chave da API Deepgram não foi encontrada. Verifique suas variáveis de ambiente.");
      }
      setStatus(ConnectionStatus.CONNECTING);
      
      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
      const deepgram = createClient(process.env.DEEPGRAM_API_KEY);
      
      const systemInstruction = `
        Você é o "Blue Ray Agent", um robô analista de Price Action para Pocket Option.
        Você está em uma chamada LIVE, observando a tela do trader em tempo real.
        
        COMPORTAMENTO:
        - Analise o gráfico continuamente.
        - Fale proativamente: Se vir uma vela de força, um padrão de reversão (martelo, engolfo), um toque em suporte/resistência, ou qualquer movimento relevante, narre o que está acontecendo IMEDIATAMENTE.
        - Não espere o usuário falar. Sua função é ser um mentor ativo que aponta oportunidades e padrões.
        - Use termos técnicos: "Rompimento", "Pullback", "LTA/LTB", "Zona de exaustão".
        - Língua: Português do Brasil. Voz amigável e confiante.
        - Diga "OPORTUNIDADE DETECTADA" antes de sugerir uma Compra ou Venda.
      `;

      if (!outputAudioContextRef.current) {
        outputAudioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
        audioOutputNodeRef.current = outputAudioContextRef.current.createGain();
        audioOutputNodeRef.current.connect(outputAudioContextRef.current.destination);
      }
      if (!inputAudioContextRef.current) {
        inputAudioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 16000 });
      }

      let mediaStream: MediaStream;
      if (mode === 'screen') {
        mediaStream = await navigator.mediaDevices.getDisplayMedia({ video: { frameRate: 15 }, audio: false });
      } else {
        mediaStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" }, audio: false });
      }

      const micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      micStreamRef.current = micStream;

      const deepgramConnection = deepgram.listen.live({
        language: 'pt-BR',
        model: 'nova-2',
        smart_format: true,
        interim_results: true,
      });
      deepgramConnectionRef.current = deepgramConnection;

      deepgramConnection.on('transcript', (data) => {
        const transcript = data.channel.alternatives[0].transcript;
        if (!transcript) return;
        setUserTranscript(transcript);
        if (data.is_final && data.speech_final) {
          addMessage('user', transcript);
          setUserTranscript('');
        }
      });
      deepgramConnection.on('error', (e) => {
        console.error('Deepgram error:', e);
        setErrorMsg('Erro na transcrição de voz. Verifique a chave da API Deepgram.');
      });


      const sessionPromise = ai.live.connect({
        model: GEMINI_MODEL,
        config: {
          responseModalities: [Modality.AUDIO],
          speechConfig: { voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Puck' } } },
          systemInstruction,
          outputAudioTranscription: {},
        },
        callbacks: {
          onopen: () => {
            setStatus(ConnectionStatus.CONNECTED);
            setIsCapturing(true);
          },
          onmessage: async (message: LiveServerMessage) => {
            const base64Audio = message.serverContent?.modelTurn?.parts[0]?.inlineData?.data;
            if (base64Audio && outputAudioContextRef.current && audioOutputNodeRef.current) {
              setIsSpeaking(true);
              const ctx = outputAudioContextRef.current;
              nextStartTimeRef.current = Math.max(nextStartTimeRef.current, ctx.currentTime);
              const audioBuffer = await decodeAudioData(decode(base64Audio), ctx, 24000, 1);
              const source = ctx.createBufferSource();
              source.buffer = audioBuffer;
              source.connect(audioOutputNodeRef.current);
              source.addEventListener('ended', () => {
                  sourcesRef.current.delete(source);
                  if (sourcesRef.current.size === 0) setIsSpeaking(false);
              });
              source.start(nextStartTimeRef.current);
              nextStartTimeRef.current += audioBuffer.duration;
              sourcesRef.current.add(source);
            }

            if (message.serverContent?.outputTranscription) addMessage('assistant', message.serverContent.outputTranscription.text);
            if (message.serverContent?.interrupted) {
              sourcesRef.current.forEach(s => { try { s.stop(); } catch(e){} });
              sourcesRef.current.clear();
              nextStartTimeRef.current = 0;
              setIsSpeaking(false);
            }
          },
          onerror: (e) => {
            setStatus(ConnectionStatus.ERROR);
            setErrorMsg("Conexão Gemini interrompida.");
            stopCapture();
          },
          onclose: () => stopCapture()
        }
      });
      
      sessionPromiseRef.current = sessionPromise;
      sessionRef.current = await sessionPromise;

      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
        videoRef.current.play();
      }

      const micSource = inputAudioContextRef.current!.createMediaStreamSource(micStream);
      const processor = inputAudioContextRef.current!.createScriptProcessor(4096, 1, 1);
      processor.onaudioprocess = (e) => {
        const inputData = e.inputBuffer.getChannelData(0);
        // Send to Gemini
        sessionPromise.then(s => s.sendRealtimeInput({ media: createAudioBlob(inputData) }));
        // Send to Deepgram
        if (deepgramConnectionRef.current?.getReadyState() === 1) { // 1 = OPEN
            deepgramConnectionRef.current.send(inputData);
        }
      };
      micSource.connect(processor);
      processor.connect(inputAudioContextRef.current!.destination);

      captureIntervalRef.current = window.setInterval(async () => {
          if (!videoRef.current || !canvasRef.current || !sessionPromiseRef.current) return;
          const canvas = canvasRef.current;
          const video = videoRef.current;
          const ctx = canvas.getContext('2d');
          if (ctx && video.videoWidth > 0) {
              canvas.width = 640;
              canvas.height = (video.videoHeight / video.videoWidth) * 640;
              ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
              canvas.toBlob(async (blob) => {
                  if (blob) {
                      const base64Data = await blobToBase64(blob);
                      const session = await sessionPromiseRef.current;
                      session.sendRealtimeInput({ media: { data: base64Data, mimeType: 'image/jpeg' } });
                  }
              }, 'image/jpeg', JPEG_QUALITY);
          }
      }, 1000 / FRAME_RATE);

    } catch (err: any) {
      setErrorMsg(err.message);
      setStatus(ConnectionStatus.ERROR);
      stopCapture();
    }
  };
  
  return (
    <div className="fixed inset-0 bg-black text-white flex flex-col font-sans overflow-hidden">
      <header className="absolute top-0 inset-x-0 p-6 flex justify-between items-center z-50 bg-gradient-to-b from-black/80 to-transparent">
        <div className="flex items-center gap-3">
          <div className={`w-3 h-3 rounded-full ${status === ConnectionStatus.CONNECTED ? 'bg-green-500 shadow-[0_0_10px_#22c55e]' : 'bg-red-500 animate-pulse'}`}></div>
          <span className="text-[11px] font-black tracking-[0.2em] uppercase text-white/80">
            {isCapturing ? 'ANÁLISE EM TEMPO REAL' : 'SISTEMA OFFLINE'}
          </span>
        </div>
        <button 
            disabled={isCapturing}
            onClick={() => setMode(mode === 'screen' ? 'camera' : 'screen')}
            className={`px-4 py-1.5 rounded-full text-[10px] font-bold border transition-all ${mode === 'screen' ? 'bg-blue-600 border-blue-400' : 'bg-white/5 border-white/10'} disabled:opacity-50`}
        >
            {mode === 'screen' ? 'MODO TELA' : 'MODO CÂMERA'}
        </button>
      </header>
      
      <div className="relative flex-1 flex flex-col justify-center items-center z-10 p-6 overflow-hidden">
        <video ref={videoRef} className={`w-full h-full object-contain transition-opacity duration-500 ${isCapturing ? 'opacity-100' : 'opacity-0'}`} muted playsInline />
        <canvas ref={canvasRef} className="hidden" />

        {!isCapturing && (
            <div className="text-center">
                <div className="w-40 h-40 rounded-full bg-gradient-to-br from-blue-900/40 to-black border-2 border-white/10 flex items-center justify-center mx-auto">
                    <svg className="w-14 h-14 text-white/20" fill="currentColor" viewBox="0 0 20 20"><path fillRule="evenodd" d="M7 4a3 3 0 016 0v4a3 3 0 11-6 0V4zm4 10.93A7.001 7.001 0 0017 8a1 1 0 10-2 0A5 5 0 015 8a1 1 0 00-2 0 7.001 7.001 0 005.93 6.93V17H6a1 1 0 100 2h8a1 1 0 100-2h-3v-2.07z" clipRule="evenodd" /></svg>
                </div>
                <h1 className="mt-8 text-2xl font-black tracking-tighter text-white">BLUE RAY AGENT</h1>
                <p className="text-[10px] uppercase tracking-[0.4em] text-white/40 font-bold">Trading Intelligent System</p>
            </div>
        )}

        <div className="absolute bottom-4 left-6 right-6 h-32 flex flex-col-reverse gap-2 overflow-hidden pointer-events-none bg-gradient-to-t from-black/50 to-transparent p-4">
            {userTranscript && (
                <div className="text-left text-sm font-medium text-white/70 animate-pulse">
                    <span className="font-bold mr-2">Você:</span>
                    {userTranscript}
                </div>
            )}
            {messages.map((m, i) => (
                <div key={i} className={`text-left text-sm font-medium transition-all duration-500 ${m.role === 'assistant' ? 'text-blue-300' : 'text-white/40'}`}>
                    <span className="font-bold mr-2">{m.role === 'assistant' ? 'Blue Ray:' : 'Você:'}</span>
                    {m.text}
                </div>
            ))}
        </div>
      </div>
      
      {errorMsg && (
        <div className="absolute bottom-32 left-6 right-6 p-4 bg-red-600 rounded-2xl flex items-start gap-3 shadow-2xl z-20 animate-in slide-in-from-bottom duration-300">
            <span className="text-xl">⚠️</span>
            <p className="text-sm font-bold text-white leading-tight">{errorMsg}</p>
        </div>
      )}

      <footer className="p-6 pb-8 bg-black/30 backdrop-blur-sm border-t border-white/5 z-20 flex flex-col items-center">
        {!isCapturing ? (
            <>
                <button onClick={startAnalysis} className="group relative w-24 h-24 flex items-center justify-center transition-all active:scale-90">
                    <div className="absolute inset-0 bg-blue-600 rounded-full blur-xl opacity-40 group-hover:opacity-60 transition-opacity"></div>
                    <div className="relative w-full h-full bg-blue-600 rounded-full flex items-center justify-center border-4 border-blue-400 shadow-2xl">
                        <svg className="w-10 h-10 text-white" fill="currentColor" viewBox="0 0 20 20"><path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z" clipRule="evenodd" /></svg>
                    </div>
                </button>
                <p className="mt-4 text-[9px] font-black uppercase text-white/30 tracking-[0.3em]">Ativar Mentoria</p>
            </>
        ) : (
             <>
                <button onClick={stopCapture} className="w-20 h-20 bg-red-500/20 rounded-full flex items-center justify-center border-2 border-red-400/50 active:scale-95 transition-all">
                    <div className="w-7 h-7 bg-red-500 rounded-md animate-pulse"></div>
                </button>
                <p className="mt-4 text-[9px] font-black uppercase text-white/30 tracking-[0.3em]">Parar Análise</p>
             </>
        )}
      </footer>
    </div>
  );
};

export default App;
