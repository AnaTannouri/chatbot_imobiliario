import os
import re
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from openai import OpenAI
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters

# === CARREGA VARIÁVEIS DO .env ===
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME")

# === INICIALIZA CLIENTES ===
client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# === MODELO DE EMBEDDINGS ===
modelo = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

# === OUTRAS CONFIGS ===
LINK_PESQUISA = "📝 **Participe da nossa pesquisa de satisfação:** [Clique aqui](https://docs.google.com/forms/d/e/1FAIpQLSeVz6AxoTvTB4pHxgKU-sso3GAUgA_irTBu4LGpkQgcTAThsQ/viewform?usp=header)"

PALAVRAS_ATENDENTE = [
    "atendente", "pessoa", "suporte", "falar com alguém", "ajuda humana", "representante"
]

MENSAGEM_BOAS_VINDAS = (
    "Olá! 👋 Sou o Chatbot Imobiliário!\n\n"
    "📘 Aviso de Privacidade (LGPD): As informações fornecidas serão tratadas conforme a "
    "Lei Geral de Proteção de Dados (Lei nº 13.709/2018), usadas apenas para fins de atendimento e consulta.\n\n"
    "Como posso lhe ajudar?\n\n"
    f"{LINK_PESQUISA}"
)

# === /start ===
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(MENSAGEM_BOAS_VINDAS, parse_mode="Markdown")

# === /ajuda ===
async def ajuda(update: Update, context: ContextTypes.DEFAULT_TYPE):
    exemplos = (
        "📌 Exemplos de perguntas:\n"
        "- Como financiar um imóvel?\n"
        "- Quais documentos preciso?\n"
        "- O que é o Minha Casa Minha Vida?\n"
        "- Posso comprar com FGTS?\n"
    )
    await update.message.reply_text(exemplos)

# === Mensagens comuns ===
async def responder(update: Update, context: ContextTypes.DEFAULT_TYPE):
    pergunta = update.message.text.lower().strip()
    print(f"📩 Mensagem recebida: {pergunta}")

    if any(re.search(p, pergunta) for p in PALAVRAS_ATENDENTE):
        msg = "Certo! Vou encaminhar sua solicitação para um de nossos atendentes humanos. 👩‍💼"
        await update.message.reply_text(f"{msg}\n\n{LINK_PESQUISA}", parse_mode="Markdown")
        return

    try:
        vetor = modelo.encode(pergunta).tolist()
        resultados = index.query(vector=vetor, top_k=5, include_metadata=True)
    except Exception as e:
        print(f"❌ Erro ao consultar o Pinecone: {e}")
        await update.message.reply_text("Desculpe, ocorreu um erro temporário. Tente novamente em alguns instantes.")
        return

    # Ordena e filtra por relevância
    matches = sorted(
        [m for m in resultados.get("matches", []) if m["score"] >= 0.40],
        key=lambda x: x["score"],
        reverse=True
    )

    # Limita o tamanho do contexto
    MAX_TOKENS = 1500
    contexto, tokens_total = "", 0
    for m in matches:
        trecho = m["metadata"]["resposta"]
        tokens = len(trecho.split())
        if tokens_total + tokens > MAX_TOKENS:
            break
        contexto += trecho + "\n\n"
        tokens_total += tokens

    if not contexto:
        respostas = {
            "casa": "Você gostaria de **comprar** ou **alugar** uma casa?",
            "imovel": "Você gostaria de **comprar** ou **alugar** um imóvel?",
            "comprar": "Você quer **comprar um imóvel** por **financiamento** ou **à vista**?",
            "financiamento": "Posso te explicar sobre **financiamento imobiliário**! Quer saber as **condições** ou o **passo a passo**?",
            "documento": "Você quer saber quais **documentos** são necessários para **comprar** ou **vender** um imóvel?",
            "programa": "Está se referindo aos **programas habitacionais**, como o **Minha Casa, Minha Vida**?"
        }
        for chave, resp in respostas.items():
            if chave in pergunta:
                await update.message.reply_text(f"{resp}\n\n{LINK_PESQUISA}", parse_mode="Markdown")
                return
        await update.message.reply_text(
            f"Desculpe, não encontrei essa informação na base. 🤝\n\n{LINK_PESQUISA}",
            parse_mode="Markdown"
        )
        return

    prompt = f"""
    Você é um assistente imobiliário especializado.
    Responda **somente** com base nas informações abaixo.

    --- CONTEXTO ---
    {contexto}

    --- PERGUNTA ---
    {pergunta}
    """

    try:
        resposta = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Você é um assistente imobiliário confiável, educado e objetivo."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )

        resposta_final = resposta.choices[0].message.content.strip()
        await update.message.reply_text(f"{resposta_final}\n\n{LINK_PESQUISA}", parse_mode="Markdown")

    except Exception as e:
        print(f"❌ Erro ao gerar resposta: {e}")
        await update.message.reply_text(f"Desculpe, houve um erro ao gerar a resposta.\n\n{LINK_PESQUISA}", parse_mode="Markdown")

# === Inicialização com webhook (Ngrok) ===
def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("ajuda", ajuda))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, responder))

    print("🚀 Chatbot rodando com webhook!")

    # Atualize com seu link do Ngrok abaixo:
    NGROK_URL = "https://autogenetic-yer-unrepealed.ngrok-free.dev"

    app.run_webhook(
    listen="0.0.0.0",
    port=8080,
    webhook_url=f"{NGROK_URL}/"
    )


if __name__ == "__main__":
    main()
