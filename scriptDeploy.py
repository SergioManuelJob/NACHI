import asyncio
import tensorflow as tf
from tensorflow.keras.models import load_model
import logging
from telegram import Update
from telegram.ext import filters, ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler
import os
from dotenv import load_dotenv
load_dotenv()

IMG_SIZE = (180, 180)
model = load_model("NACHI/Modelo Final/ResnetModel.h5")

# Esta parte nos ayuda a debugguear cualquier problema
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Funcion que ocurre cuando se hace /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Hi I'm NACHI a Neural Architecture who's purpose is to recognize photos of butterflies as of right now in it's first version, please talk to me in case you need help identifying different butterflies! \n\nTo start predicting, just send me a photo of a pretty butterfly! \n\nFor your interest, this are the different names of the species of butterflies I have been trained with: '88 AN', 'ADMIRAL RED', 'ADONIS', 'AFRICAN GIANT SWALLOWTAIL', 'AMERICAN SNOOT', 'APPOLLO', 'ATALA', 'AWL BANDED COMMON', 'BANDED GOLD', 'BANDED HELICONIAN ORANGE', 'BANDED PEACOCK', 'BARRED FLASHER TWO', 'BECKERS WHITE', 'BIRDWING CAIRNS', 'BLACK HAIRSTREAK', 'BLUE CROW SPOTTED', 'BLUE MORPHO', 'BROWN SIPROETA', 'CABBAGE WHITE', 'CATTLEHEART CELLED GREEN', 'CHECQUERED SKIPPER', 'CHESTNUT', 'CLEOPATRA', 'CLOAK MOURNING', 'CLODIUS PARNASSIAN', 'CLOUDED SULPHUR', 'COMA EASTERN', 'COMMON WOOD-NYMPH', 'COPPER PURPLISH', 'COPPER TAIL', 'CRACKER RED', 'CRECENT', 'CRIMSON PATCH', 'DANAID EGGFLY', 'DAPPLE EASTERN WHITE', 'DOGFACE SOUTHERN', 'EASTERN ELFIN PINE', 'EGGFLY GREAT', 'ELBOWED PIERROT', 'GREAT JAY', 'GREY HAIRSTREAK', 'HAIRSTREAK PURPLE', 'INDRA SWALLOW', 'IPHICLUS SISTER', 'JULIA', 'KITE PAPER', 'LADY PAINTED', 'LARGE MARBLE', 'LEAFWING TROPICAL', 'LONG WING ZEBRA', 'MALACHITE', 'MANGROVE SKIPPER', 'MARK QUESTION', 'MESTRA', 'METALMARK', 'MILBERTS TORTOISESHELL', 'MONARCH', 'OAKLEAF ORANGE', 'ORANGE SLEEPY', 'ORANGE TIP', 'ORCHARD SWALLOW', 'PEACOCK', 'PINE WHITE', 'PIPEVINE SWALLOW', 'POPINJAY', 'POSTMAN RED', 'PURPLE RED SPOTTED', 'QUEEN STRAITED', 'SATYR WOOD', 'SCARCE SWALLOW', 'SILVER SKIPPER SPOT', 'SOOTYWING', 'SWALLOW TAIL YELLOW', 'ULYSES', 'VICEROY'.")

async def respond(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="If you want to start predicting, send me a photo of a butterfly!")

async def predict(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(context.args)
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Let's predict that butterfly image!")

async def predict(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Verifica que el mensaje contenga una foto
    if update.message.photo:
        # Obtén la foto de mayor resolución
        photo = update.message.photo[-1]
        file = await context.bot.get_file(photo.file_id)
        file_path = f"downloads/{photo.file_id}.jpg"
        
        # Descarga la foto
        await file.download_to_drive(file_path)
        
        # Preprocesa la imagen
        img = image.load_img(file_path, target_size=IMG_SIZE)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Añade una dimensión extra para el batch

        # Normaliza la imagen (ajusta esto según los requisitos de tu modelo)
        img_array /= 255.0

        # Realiza la predicción
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)

        # Envía la respuesta al usuario
        await context.bot.send_message(chat_id=update.effective_chat.id, text=f"La predicción es: {predicted_class[0]}")

        # Limpia el archivo descargado
        os.remove(file_path)
    else:
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Por favor, envíame una foto para hacer la predicción.")


if __name__ == '__main__':
    application = ApplicationBuilder().token(os.getenv("API_KEY")).build()
    
    start_handler = CommandHandler('start', start)
    application.add_handler(start_handler)

    echo_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), respond)
    application.add_handler(echo_handler)
    
    predict_handler = MessageHandler(filters.PHOTO, predict)
    application.add_handler(predict_handler)

    application.run_polling()