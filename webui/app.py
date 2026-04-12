#!/usr/bin/env python3
import os
import sys
import json
import threading
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
import gradio as gr
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from webui.ensiRNA_wrapper import predict_ensiRNA, ENSIRNAWrapper
from webui.ensiRNA_mod_wrapper import predict_ensiRNA_mod, ENSIRNAModWrapper


class Task(BaseModel):
    id: str
    model_type: str
    status: str
    created_at: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class TaskManager:
    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.lock = threading.Lock()
        self._save_file = "tasks.json"
        self._load_tasks()

    def _load_tasks(self):
        if os.path.exists(self._save_file):
            try:
                with open(self._save_file, "r") as f:
                    tasks_data = json.load(f)
                    for task_data in tasks_data:
                        self.tasks[task_data["id"]] = Task(**task_data)
            except Exception:
                pass

    def _save_tasks(self):
        with self.lock:
            try:
                with open(self._save_file, "w") as f:
                    json.dump(
                        [task.model_dump() for task in self.tasks.values()], f, indent=2
                    )
            except Exception:
                pass

    def add_task(self, model_type: str) -> str:
        task_id = str(uuid.uuid4())[:8]
        with self.lock:
            self.tasks[task_id] = Task(
                id=task_id,
                model_type=model_type,
                status="pending",
                created_at=datetime.now().isoformat(),
            )
        self._save_tasks()
        return task_id

    def update_task(
        self,
        task_id: str,
        status: str,
        result: Optional[Dict] = None,
        error: Optional[str] = None,
    ):
        with self.lock:
            if task_id in self.tasks:
                self.tasks[task_id].status = status
                if result is not None:
                    self.tasks[task_id].result = result
                if error is not None:
                    self.tasks[task_id].error = error
        self._save_tasks()

    def get_tasks(self) -> List[Task]:
        with self.lock:
            return list(self.tasks.values())

    def get_task(self, task_id: str) -> Optional[Task]:
        with self.lock:
            return self.tasks.get(task_id)

    def clear_completed(self):
        with self.lock:
            completed_ids = [
                tid for tid, task in self.tasks.items() if task.status == "completed"
            ]
            for tid in completed_ids:
                del self.tasks[tid]
        self._save_tasks()


task_manager = TaskManager()

ensiRNA_model = None
ensiRNA_mod_model = None


def get_ensiRNA_model():
    global ensiRNA_model
    if ensiRNA_model is None:
        ensiRNA_model = ENSIRNAWrapper()
    return ensiRNA_model


def get_ensiRNA_mod_model():
    global ensiRNA_mod_model
    if ensiRNA_mod_model is None:
        ensiRNA_mod_model = ENSIRNAModWrapper()
    return ensiRNA_mod_model


def run_ensiRNA_task(
    task_id: str,
    siRNA_id: str,
    sense_seq: str,
    anti_seq: str,
    mRNA_seq: str,
    position: int,
    ckpt_path: str,
    gpu: int,
):
    try:
        task_manager.update_task(task_id, "running")
        wrapper = get_ensiRNA_model()
        result = wrapper.predict(
            siRNA_id,
            sense_seq,
            anti_seq,
            mRNA_seq,
            position,
            ckpt_path=ckpt_path,
            gpu=gpu,
        )
        if "error" in result and result["error"]:
            task_manager.update_task(task_id, "failed", error=result["error"])
        else:
            task_manager.update_task(task_id, "completed", result=result)
    except Exception as e:
        task_manager.update_task(task_id, "failed", error=str(e))


def run_ensiRNA_mod_task(
    task_id: str,
    siRNA_id: str,
    sense_seq: str,
    anti_seq: str,
    sense_mods: List[str],
    anti_mods: List[str],
    ckpt_path: str,
    gpu: int,
):
    try:
        task_manager.update_task(task_id, "running")

        sense_mod_dict = {}
        for mod in sense_mods:
            if mod and ":" in mod:
                mod_type, mod_pos = mod.split(":")
                sense_mod_dict[mod_type] = mod_pos

        anti_mod_dict = {}
        for mod in anti_mods:
            if mod and ":" in mod:
                mod_type, mod_pos = mod.split(":")
                anti_mod_dict[mod_type] = mod_pos

        wrapper = get_ensiRNA_mod_model()
        result = wrapper.predict(
            siRNA_id,
            sense_seq,
            anti_seq,
            sense_mod_dict,
            anti_mod_dict,
            ckpt_path=ckpt_path,
            gpu=gpu,
        )

        if "error" in result and result["error"]:
            task_manager.update_task(task_id, "failed", error=result["error"])
        else:
            task_manager.update_task(task_id, "completed", result=result)
    except Exception as e:
        task_manager.update_task(task_id, "failed", error=str(e))


def submit_ensiRNA(
    siRNA_id: str,
    sense_seq: str,
    anti_seq: str,
    mRNA_seq: str,
    position: int,
    ckpt_path: str,
    gpu: int,
):
    if not siRNA_id:
        return "Error: Please enter siRNA ID", ""
    if not sense_seq:
        return "Error: Please enter sense sequence", ""
    if not anti_seq:
        return "Error: Please enter anti-sense sequence", ""

    task_id = task_manager.add_task("ENsiRNA")
    thread = threading.Thread(
        target=run_ensiRNA_task,
        args=(
            task_id,
            siRNA_id,
            sense_seq,
            anti_seq,
            mRNA_seq,
            position,
            ckpt_path,
            gpu,
        ),
    )
    thread.start()

    return f"Task submitted! Task ID: {task_id}", task_id


def submit_ensiRNA_mod(
    siRNA_id: str,
    sense_seq: str,
    anti_seq: str,
    sense_mods: List[str],
    anti_mods: List[str],
    ckpt_path: str,
    gpu: int,
):
    if not siRNA_id:
        return "Error: Please enter siRNA ID", ""
    if not sense_seq:
        return "Error: Please enter sense sequence", ""
    if not anti_seq:
        return "Error: Please enter anti-sense sequence", ""

    task_id = task_manager.add_task("ENsiRNA-Mod")
    thread = threading.Thread(
        target=run_ensiRNA_mod_task,
        args=(
            task_id,
            siRNA_id,
            sense_seq,
            anti_seq,
            sense_mods,
            anti_mods,
            ckpt_path,
            gpu,
        ),
    )
    thread.start()

    return f"Task submitted! Task ID: {task_id}", task_id


def get_tasks_table():
    tasks = task_manager.get_tasks()
    if not tasks:
        return "No tasks yet."

    rows = []
    for task in sorted(tasks, key=lambda x: x.created_at, reverse=True):
        status_icon = {
            "pending": "⏳",
            "running": "🔄",
            "completed": "✅",
            "failed": "❌",
        }.get(task.status, "❓")

        result_str = ""
        if task.status == "completed" and task.result:
            result_val = task.result.get("result")
            if result_val is not None:
                result_str = f"{result_val:.4f}"
        elif task.status == "failed":
            result_str = task.error or "Unknown error"

        model_short = "ENsiRNA" if task.model_type == "ENsiRNA" else "ENsiRNA-Mod"

        rows.append(
            f"| {task.id} | {model_short} | {status_icon} {task.status} | {task.created_at[:19]} | {result_str} |"
        )

    header = "| Task ID | Model | Status | Created At | Result |\n|---|---|---|---|---|"
    return header + "\n" + "\n".join(rows)


def refresh_tasks():
    return get_tasks_table()


def clear_completed_tasks():
    task_manager.clear_completed()
    return get_tasks_table()


def get_result_json(task_id: str):
    task = task_manager.get_task(task_id)
    if not task:
        return "Task not found"

    if task.status == "completed" and task.result:
        return json.dumps(task.result, indent=2)
    elif task.status == "failed":
        return f"Error: {task.error}"
    elif task.status == "running":
        return "Task is still running..."
    else:
        return "Task status: " + task.status


def validate_seq(seq: str) -> bool:
    valid_bases = set("ACGUacgu")
    return all(c in valid_bases for c in seq.strip())


def create_ui():
    with gr.Blocks(title="ENsiRNA Web UI", theme=gr.themes.Soft()) as app:
        gr.Markdown("# ENsiRNA Prediction Web UI")
        gr.Markdown(
            "Submit siRNA prediction tasks for both ENsiRNA and ENsiRNA-Mod models."
        )

        with gr.Tabs():
            with gr.Tab("ENsiRNA"):
                gr.Markdown("### ENsiRNA Prediction")
                gr.Markdown("Predict siRNA efficacy without modifications.")

                with gr.Row():
                    with gr.Column():
                        siRNA_id = gr.Textbox(
                            label="siRNA ID", placeholder="e.g., Test001"
                        )
                        sense_seq = gr.Textbox(
                            label="Sense Sequence (5'->3')",
                            placeholder="e.g., CAGAAAGAGUGUCUCAUCUUA",
                        )
                        anti_seq = gr.Textbox(
                            label="Anti-sense Sequence (5'->3')",
                            placeholder="e.g., UAAGAUGAGACACUCUUUCUGGU",
                        )

                    with gr.Column():
                        mRNA_seq = gr.Textbox(
                            label="mRNA Sequence (optional)",
                            placeholder="61nt mRNA sequence",
                        )
                        position = gr.Slider(
                            minimum=0, maximum=60, value=30, step=1, label="Position"
                        )
                        ckpt_path = gr.Textbox(
                            value="ENsiRNA/pkl/checkpoint_1.ckpt",
                            label="Checkpoint Path",
                        )
                        gpu = gr.Slider(
                            minimum=-1,
                            maximum=3,
                            value=-1,
                            step=1,
                            label="GPU Device (-1 for CPU)",
                        )

                ensiRNA_submit_btn = gr.Button("Submit Task", variant="primary")
                ensiRNA_result = gr.Textbox(
                    label="Submission Result", interactive=False
                )
                ensiRNA_task_id = gr.Textbox(label="Task ID", visible=False)

                def handle_submit_ensiRNA(
                    siRNA_id, sense_seq, anti_seq, mRNA_seq, position, ckpt_path, gpu
                ):
                    if not validate_seq(sense_seq) or not validate_seq(anti_seq):
                        return "Error: Invalid sequence. Use only A, C, G, U.", ""
                    return submit_ensiRNA(
                        siRNA_id,
                        sense_seq,
                        anti_seq,
                        mRNA_seq,
                        position,
                        ckpt_path,
                        int(gpu),
                    )

                ensiRNA_submit_btn.click(
                    handle_submit_ensiRNA,
                    inputs=[
                        siRNA_id,
                        sense_seq,
                        anti_seq,
                        mRNA_seq,
                        position,
                        ckpt_path,
                        gpu,
                    ],
                    outputs=[ensiRNA_result, ensiRNA_task_id],
                )

            with gr.Tab("ENsiRNA-Mod"):
                gr.Markdown("### ENsiRNA-Mod Prediction")
                gr.Markdown("Predict siRNA efficacy with modifications.")

                with gr.Row():
                    with gr.Column():
                        siRNA_id_mod = gr.Textbox(
                            label="siRNA ID", placeholder="e.g., Test001_Mod"
                        )
                        sense_seq_mod = gr.Textbox(
                            label="Sense Sequence (5'->3')",
                            placeholder="e.g., CAGAAAGAGUGUCUCAUCUUA",
                        )
                        anti_seq_mod = gr.Textbox(
                            label="Anti-sense Sequence (5'->3')",
                            placeholder="e.g., UAAGAUGAGACACUCUUUCUGGU",
                        )

                    with gr.Column():
                        gr.Markdown("#### Sense Modifications")
                        sense_mod_1 = gr.Textbox(
                            label="Modification 1 (type:pos)",
                            placeholder="e.g., 2-Fluoro:2,3,4",
                        )
                        sense_mod_2 = gr.Textbox(
                            label="Modification 2 (type:pos)",
                            placeholder="e.g., 2-O-Methyl:1,5,7",
                        )
                        sense_mod_3 = gr.Textbox(
                            label="Modification 3 (type:pos)",
                            placeholder="e.g., Phosphorothioate:2,3",
                        )

                    with gr.Column():
                        gr.Markdown("#### Anti-sense Modifications")
                        anti_mod_1 = gr.Textbox(
                            label="Modification 1 (type:pos)",
                            placeholder="e.g., 2-Fluoro:2,3,4",
                        )
                        anti_mod_2 = gr.Textbox(
                            label="Modification 2 (type:pos)",
                            placeholder="e.g., 2-O-Methyl:1,5,7",
                        )
                        anti_mod_3 = gr.Textbox(
                            label="Modification 3 (type:pos)",
                            placeholder="e.g., Phosphorothioate:2,3",
                        )
                        ckpt_path_mod = gr.Textbox(
                            value="ENsiRNA-mod/pkl/checkpoint_1.ckpt",
                            label="Checkpoint Path",
                        )
                        gpu_mod = gr.Slider(
                            minimum=-1,
                            maximum=3,
                            value=-1,
                            step=1,
                            label="GPU Device (-1 for CPU)",
                        )

                ensiRNA_mod_submit_btn = gr.Button("Submit Task", variant="primary")
                ensiRNA_mod_result = gr.Textbox(
                    label="Submission Result", interactive=False
                )
                ensiRNA_mod_task_id = gr.Textbox(label="Task ID", visible=False)

                def handle_submit_ensiRNA_mod(
                    siRNA_id,
                    sense_seq,
                    anti_seq,
                    sm1,
                    sm2,
                    sm3,
                    am1,
                    am2,
                    am3,
                    ckpt,
                    gpu,
                ):
                    if not validate_seq(sense_seq) or not validate_seq(anti_seq):
                        return "Error: Invalid sequence. Use only A, C, G, U.", ""
                    sense_mods = [sm1, sm2, sm3]
                    anti_mods = [am1, am2, am3]
                    return submit_ensiRNA_mod(
                        siRNA_id,
                        sense_seq,
                        anti_seq,
                        sense_mods,
                        anti_mods,
                        ckpt,
                        int(gpu),
                    )

                ensiRNA_mod_submit_btn.click(
                    handle_submit_ensiRNA_mod,
                    inputs=[
                        siRNA_id_mod,
                        sense_seq_mod,
                        anti_seq_mod,
                        sense_mod_1,
                        sense_mod_2,
                        sense_mod_3,
                        anti_mod_1,
                        anti_mod_2,
                        anti_mod_3,
                        ckpt_path_mod,
                        gpu_mod,
                    ],
                    outputs=[ensiRNA_mod_result, ensiRNA_mod_task_id],
                )

            with gr.Tab("Task Management"):
                gr.Markdown("### Task Management")
                gr.Markdown("View and manage your prediction tasks.")

                with gr.Row():
                    refresh_btn = gr.Button("Refresh", variant="secondary")
                    clear_btn = gr.Button("Clear Completed", variant="secondary")

                tasks_table = gr.Markdown(get_tasks_table())

                refresh_btn.click(refresh_tasks, outputs=tasks_table)
                clear_btn.click(clear_completed_tasks, outputs=tasks_table)

                with gr.Row():
                    gr.Markdown("### View Task Result")
                    check_task_id = gr.Textbox(
                        label="Task ID", placeholder="Enter task ID to check"
                    )
                    check_btn = gr.Button("Check Result")
                    result_output = gr.Textbox(
                        label="Result", interactive=False, lines=10
                    )

                check_btn.click(
                    get_result_json, inputs=check_task_id, outputs=result_output
                )

        gr.Markdown("---")
        gr.Markdown("© 2026 ENsiRNA Web UI | Powered by Gradio")

    return app


def main():
    import argparse

    parser = argparse.ArgumentParser(description="ENsiRNA Web UI")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind")
    parser.add_argument("--share", action="store_true", help="Create share link")
    args = parser.parse_args()

    app = create_ui()
    app.launch(host=args.host, port=args.port, share=args.share)


if __name__ == "__main__":
    main()
