import { Toaster } from 'sonner';
import { TooltipProvider } from '@/components/ui/tooltip';
import { ResizablePanelGroup, ResizablePanel, ResizableHandle } from '@/components/ui/resizable';
import { Toolbar } from '@/components/Layout/Toolbar';
import { LogPanel } from '@/components/Layout/LogPanel';
import { ProblemSidebar } from '@/components/Sidebar/ProblemSidebar';
import { TreeCanvas } from '@/components/Tree/TreeCanvas';
import { DetailPanel } from '@/components/Detail/DetailPanel';
import { useSelectedNode } from '@/store/appStore';
import { usePollProblems, usePollTree, useFetchModels } from '@/api/hooks';

function App() {
  const selectedNode = useSelectedNode();

  // Start polling the backend
  usePollProblems();
  usePollTree();
  useFetchModels();

  return (
    <TooltipProvider>
      <div className="h-screen w-screen flex flex-col bg-background text-foreground overflow-hidden dark">
        {/* Toolbar */}
        <Toolbar />

        {/* Main content — resizable panels */}
        <ResizablePanelGroup orientation="horizontal" className="flex-1">
          {/* Left sidebar (fixed) */}
          <ProblemSidebar />

          {/* Center: Tree canvas */}
          <ResizablePanel defaultSize={selectedNode ? 60 : 100} minSize={20}>
            <TreeCanvas />
          </ResizablePanel>

          {/* Right: Detail panel (resizable) */}
          {selectedNode && (
            <>
              <ResizableHandle withHandle className="bg-border hover:bg-primary/20 transition-colors" />
              <ResizablePanel defaultSize={40} minSize={15} maxSize={1000} className="bg-card">
                <DetailPanel />
              </ResizablePanel>
            </>
          )}
        </ResizablePanelGroup>

        {/* Bottom: Log panel */}
        <LogPanel />
      </div>
      <Toaster theme="dark" position="bottom-right" richColors closeButton />
    </TooltipProvider>
  );
}

export default App;
