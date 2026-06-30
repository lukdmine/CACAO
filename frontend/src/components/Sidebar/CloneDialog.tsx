import { useState } from "react";
import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogFooter,
    DialogHeader,
    DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Loader2 } from "lucide-react";
import { cloneProblem } from "@/api/client";
import { refreshProblems } from "@/api/hooks";
import { toast } from "sonner";

interface CloneDialogProps {
    sourceName: string;
    open: boolean;
    onOpenChange: (open: boolean) => void;
}

export function CloneDialog({ sourceName, open, onOpenChange }: CloneDialogProps) {
    const [newName, setNewName] = useState(sourceName + "_copy");
    const [submitting, setSubmitting] = useState(false);
    const [error, setError] = useState("");

    const handleClone = async () => {
        setError("");
        if (!newName.trim()) {
            setError("Name is required");
            return;
        }
        if (!/^[a-z0-9_]+$/.test(newName)) {
            setError("Must be lowercase alphanumeric with underscores");
            return;
        }
        setSubmitting(true);
        try {
            await cloneProblem(sourceName, newName);
            toast.success(`Cloned "${sourceName}" → "${newName}"`);
            onOpenChange(false);
            await refreshProblems();
        } catch (err) {
            setError(err instanceof Error ? err.message : "Failed to clone");
        } finally {
            setSubmitting(false);
        }
    };

    return (
        <Dialog open={open} onOpenChange={(v) => { onOpenChange(v); if (!v) setError(""); }}>
            <DialogContent className="sm:max-w-md">
                <DialogHeader>
                    <DialogTitle>Clone Problem</DialogTitle>
                    <DialogDescription>
                        Create a copy of &ldquo;{sourceName}&rdquo; with a new name.
                    </DialogDescription>
                </DialogHeader>
                <div className="space-y-2">
                    <Label htmlFor="clone-name">New name</Label>
                    <Input
                        id="clone-name"
                        value={newName}
                        onChange={(e) => setNewName(e.target.value)}
                        onKeyDown={(e) => e.key === "Enter" && handleClone()}
                        placeholder="e.g. mmul_gemini_flash"
                        autoFocus
                    />
                    {error && <p className="text-sm text-destructive">{error}</p>}
                </div>
                <DialogFooter>
                    <Button variant="outline" onClick={() => onOpenChange(false)}>Cancel</Button>
                    <Button onClick={handleClone} disabled={submitting}>
                        {submitting ? <Loader2 className="h-4 w-4 animate-spin" /> : "Clone"}
                    </Button>
                </DialogFooter>
            </DialogContent>
        </Dialog>
    );
}
