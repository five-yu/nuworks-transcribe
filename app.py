# app.py (修正版)
import streamlit as st
import tempfile
import os
from faster_whisper import WhisperModel

st.set_page_config(page_title="NUWORKS 文字起こしツール v2", page_icon="📝")
st.title("📝 NUWORKS 営業通話 文字起こしアプリ (Medium版)")

# --- 設定サイドバー ---
st.sidebar.header("設定")
# モデルサイズ：mediumをデフォルトに
model_size = st.sidebar.selectbox(
    "AIモデルサイズ",
    ["base", "small", "medium", "large-v3"],
    index=2, # mediumを選択
    help="Mediumが精度と速度のバランスが良いです。"
)

# 高速化オプション
beam_size = st.sidebar.slider(
    "解析精度 (Beam Size)",
    min_value=1, max_value=5, value=1, # デフォルトを1にして高速化
    help="数値を下げると速くなりますが、少し精度が落ちる可能性があります。"
)

uploaded_file = st.file_uploader("音声ファイルをアップロード (mp3, wav, m4a)", type=["mp3", "wav", "m4a"])

if uploaded_file is not None:
    if st.button("文字起こしを開始する"):
        with st.spinner("AIが解析中です... mediumモデルのため数分かかります..."):
            try:
                # 一時ファイル作成
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name

                # モデル読み込み (CPU設定)
                model = WhisperModel(model_size, device="cpu", compute_type="int8")

                # 文字起こし実行 (beam_sizeを可変に)
                segments, info = model.transcribe(tmp_file_path, beam_size=beam_size)

                st.success(f"完了 (言語: {info.language})")

                full_text = ""
                progress_text = st.empty()
                
                for segment in segments:
                    text = segment.text
                    
                    # --- 修正点: タイムスタンプを表示しない ---
                    st.markdown(f"- {text}") # 箇条書きで表示
                    full_text += f"{text}\n"  # 時間を含まずテキストのみ追加

                st.markdown("---")
                st.subheader("結果テキスト")
                st.text_area("コピー用", full_text, height=300)

                st.download_button(
                    label="テキストをダウンロード",
                    data=full_text,
                    file_name="transcription.txt",
                    mime="text/plain"
                )

                os.remove(tmp_file_path)

            except Exception as e:
                st.error(f"エラーが発生しました: {e}")
