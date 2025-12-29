import { Component, type ErrorInfo, type ReactNode } from 'react';

interface Props {
    children?: ReactNode;
}

interface State {
    hasError: boolean;
    error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
    public state: State = {
        hasError: false,
        error: null
    };

    public static getDerivedStateFromError(error: Error): State {
        return { hasError: true, error };
    }

    public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
        console.error("Zenith Error:", error, errorInfo);
    }

    public render() {
        if (this.state.hasError) {
            return (
                <div className="w-screen h-screen bg-black flex flex-col justify-center items-center text-center p-8 text-zinc-400">
                    <div className="mb-6 text-6xl opacity-50">🧘</div>
                    <h1 className="text-2xl font-bold text-white mb-2 uppercase tracking-widest">Meditation Interrupted</h1>
                    <p className="max-w-md mb-8">The flow was disrupted by an unexpected disturbance.</p>
                    <div className="bg-zinc-900 p-4 rounded border border-zinc-800 font-mono text-xs text-red-400 mb-8 max-w-2xl overflow-auto text-left">
                        {this.state.error?.toString()}
                    </div>
                    <button
                        onClick={() => window.location.reload()}
                        className="px-6 py-2 bg-white text-black font-bold uppercase tracking-widest hover:bg-zinc-200 transition-colors rounded"
                    >
                        Reset Flow
                    </button>
                </div>
            );
        }

        return this.props.children;
    }
}
