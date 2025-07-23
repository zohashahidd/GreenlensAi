import gradio as gr
from uuid import uuid4
from backend.chatbot import respond, start_new_conversation, handle_upload

with gr.Blocks(css="""...your custom CSS...""") as demo:

    gr.Markdown("""...your Markdown header...""")

    with gr.Row():
        with gr.Column(scale=1, min_width=220, elem_classes="sidebar"):
            gr.Markdown("### 📓 Sessions")
            new_chat_btn = gr.Button("➕ New Chat")
            pdf_upload = gr.File(label="📄 Upload ESG PDF", file_types=[".pdf"])
            upload_status = gr.Textbox(label="Upload Status", interactive=False)

        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="💬 Chat with GreenLens")
            user_input = gr.Textbox(placeholder="Ask about emissions, scores, ESG goals...", label="Your Question")
            send_btn = gr.Button("🌱 Ask")
            hidden_session_id = gr.State(str(uuid4()))

    send_btn.click(fn=respond, inputs=[user_input, hidden_session_id], outputs=[user_input, chatbot])
    new_chat_btn.click(fn=start_new_conversation, outputs=[hidden_session_id, chatbot])
    pdf_upload.change(fn=handle_upload, inputs=[pdf_upload, hidden_session_id], outputs=upload_status)

demo.launch()
