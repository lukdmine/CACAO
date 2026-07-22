import { useEffect, useRef, useState, useCallback } from 'react';
import { useAppStore } from '@/store/appStore';
import { fetchLogs } from '@/api/client';
import { Terminal, ChevronDown, ChevronUp } from 'lucide-react';

const LOG_POLL_MS = 5000;
const MIN_HEIGHT = 80;
const MAX_HEIGHT = 600;
const DEFAULT_HEIGHT = 256;
// How close to the bottom still counts as "following the tail".
const BOTTOM_SLACK_PX = 24;

export function LogPanel() {
    const activeProblem = useAppStore((s) => s.activeProblem);
    const connected = useAppStore((s) => s.connected);
    const [log, setLog] = useState('');
    const [totalLines, setTotalLines] = useState(0);
    const [expanded, setExpanded] = useState(false);
    const [height, setHeight] = useState(DEFAULT_HEIGHT);
    const scrollRef = useRef<HTMLDivElement>(null);
    const dragging = useRef(false);
    // Whether to keep following new output. A ref, not state: it changes on every scroll
    // event and must not re-render the log.
    const stickToBottom = useRef(true);

    useEffect(() => {
        if (!activeProblem || !connected) return;
        const problem = activeProblem;

        let cancelled = false;

        async function poll() {
            try {
                const data = await fetchLogs(problem);
                if (!cancelled) {
                    setLog(data.log);
                    setTotalLines(data.lines);
                }
            } catch {
                // Backend not available
            }
        }

        poll();
        const interval = setInterval(poll, LOG_POLL_MS);
        return () => { cancelled = true; clearInterval(interval); };
    }, [activeProblem, connected]);

    // Follow the tail as new log data arrives — but only while the reader is already at
    // the bottom. Scrolling up is how you read what just went past, and a poll every few
    // seconds used to yank you straight back down.
    useEffect(() => {
        const el = scrollRef.current;
        if (el && expanded && stickToBottom.current) {
            el.scrollTop = el.scrollHeight;
        }
    }, [log, expanded]);

    // Re-attach to the tail when the reader scrolls back down to it.
    const onScroll = useCallback(() => {
        const el = scrollRef.current;
        if (!el) return;
        // Slack, because scrollTop is fractional at the bottom on HiDPI/zoom and would
        // otherwise never compare equal.
        stickToBottom.current =
            el.scrollHeight - el.scrollTop - el.clientHeight < BOTTOM_SLACK_PX;
    }, []);

    const onMouseDown = useCallback((e: React.MouseEvent) => {
        e.preventDefault();
        dragging.current = true;
        const startY = e.clientY;
        const startH = height;

        const onMouseMove = (ev: MouseEvent) => {
            if (!dragging.current) return;
            const newH = Math.min(MAX_HEIGHT, Math.max(MIN_HEIGHT, startH + (startY - ev.clientY)));
            setHeight(newH);
        };
        const onMouseUp = () => {
            dragging.current = false;
            document.removeEventListener('mousemove', onMouseMove);
            document.removeEventListener('mouseup', onMouseUp);
        };
        document.addEventListener('mousemove', onMouseMove);
        document.addEventListener('mouseup', onMouseUp);
    }, [height]);

    if (!connected || !log) return null;

    return (
        <div className="border-t bg-card flex flex-col shrink-0" style={{ height: expanded ? height : 32 }}>
            {/* Resize handle */}
            {expanded && (
                <div
                    onMouseDown={onMouseDown}
                    className="h-1 shrink-0 cursor-row-resize hover:bg-primary/30 active:bg-primary/50 transition-colors"
                />
            )}

            {/* Header bar */}
            <button
                onClick={() => setExpanded(!expanded)}
                className="h-8 shrink-0 flex items-center gap-2 px-3 text-xs text-muted-foreground hover:text-foreground transition-colors cursor-pointer"
            >
                <Terminal size={13} />
                <span className="font-medium">Output</span>
                <span className="text-[10px] opacity-60">{totalLines} lines</span>
                <div className="flex-1" />
                {expanded ? <ChevronDown size={13} /> : <ChevronUp size={13} />}
            </button>

            {/* Log content */}
            {expanded && (
                <div ref={scrollRef} onScroll={onScroll} className="flex-1 overflow-auto px-3 pb-2">
                    <pre className="text-[11px] font-mono text-zinc-300 whitespace-pre-wrap leading-relaxed">{log}</pre>
                </div>
            )}
        </div>
    );
}
