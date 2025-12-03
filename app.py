# app.py

# --- 1. ライブラリのインポート ---
# streamlit: Webアプリの画面を作るためのライブラリ
import streamlit as st
# tempfile: アップロードされた一時ファイルを扱うための標準ライブラリ
import tempfile
# os: ファイルパスの操作やファイルの削除を行うための標準ライブラリ
import os
# faster_whisper: 高速な文字起こしを行うAIライブラリ
from faster_whisper import WhisperModel

# --- 2. 画面の基本設定 ---
# ページの設定（タイトルやアイコン）
st.set_page_config(page_title="NUWORKS 文字起こしツール", page_icon="📝")
st.title("📝 NUWORKS 営業通話 文字起こしアプリ")
st.write("mp3ファイルをアップロードすると、AIが自動で文字起こしを行います。")

# --- 3. サイドバーの設定（AIモデルの選択など） ---
# サイドバーに設定項目を置くことで、メイン画面をすっきりさせます
model_size = st.sidebar.selectbox(
    "AIモデルのサイズを選択",
    ["base", "small", "medium", "large-v3"],
    index=1, # デフォルトは 'small'
    help="サイズが大きいほど精度は上がりますが、処理時間が長くなります。"
)

# --- 4. ファイルアップロード機能 ---
uploaded_file = st.file_uploader("音声ファイルをアップロード (mp3, wav, m4a)", type=["mp3", "wav", "m4a"])

# --- 5. 文字起こし処理の実行 ---
# ファイルがアップロードされ、かつボタンが押されたら処理開始
if uploaded_file is not None:
    if st.button("文字起こしを開始する"):
        
        # 処理中のスピナー（ぐるぐる）を表示
        with st.spinner("AIが音声を解析中です...しばらくお待ちください..."):
            
            try:
                # --- A. 一時ファイルの作成 ---
                # faster-whisperは「ファイルパス」を必要とするため、
                # メモリ上のデータを一度PCの一時フォルダに保存します。
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name

                # --- B. AIモデルの読み込み ---
                # CPUで動く設定にしています（Web上の無料サーバーはGPUがないため）
                # int8は計算を軽量化する設定です
                model = WhisperModel(model_size, device="cpu", compute_type="int8")

                # --- C. 文字起こしの実行 ---
                # segments: 文字起こしされた文章の断片
                # info: 言語などの情報
                segments, info = model.transcribe(tmp_file_path, beam_size=5)

                st.success(f"完了しました！ (検出言語: {info.language})")

                # --- D. 結果の整形と表示 ---
                # 文字起こし結果を結合して一つのテキストにする
                full_text = ""
                
                # プログレスバー（進捗バー）の表示用
                progress_text = st.empty()
                
                for segment in segments:
                    # タイムスタンプ（開始時間）を見やすく整形
                    start_time = f"{segment.start:.1f}秒"
                    text = segment.text
                    
                    # 画面に逐次表示（チャットのようにポンポン出てくる）
                    st.markdown(f"**[{start_time}]** {text}")
                    
                    # 保存用テキストに追加
                    full_text += f"[{start_time}] {text}\n"

                # --- E. 全文の表示とダウンロード ---
                st.markdown("---") # 区切り線
                st.subheader("全体の結果")
                st.text_area("コピー用", full_text, height=300)

                # テキストファイルとしてダウンロードするボタン
                st.download_button(
                    label="テキストファイルとしてダウンロード",
                    data=full_text,
                    file_name="transcription.txt",
                    mime="text/plain"
                )

                # --- F. お掃除 ---
                # 使い終わった一時ファイルを削除する
                os.remove(tmp_file_path)

            except Exception as e:
                st.error(f"エラーが発生しました: {e}")