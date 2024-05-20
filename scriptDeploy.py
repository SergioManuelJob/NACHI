import asyncio
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import logging
from telegram import Update
from telegram.ext import filters, ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler
import os
from PIL import Image
from dotenv import load_dotenv
load_dotenv()

IMG_SIZE = (180, 180)
model = load_model("NACHI/ModeloFinal/ResnetModel.h5")
dict_classes = {0: '88 AN', 1: 'ADMIRAL RED', 2: 'ADONIS', 3: 'AFRICAN GIANT SWALLOWTAIL', 4: 'AMERICAN SNOOT', 5: 'APPOLLO', 6: 'ATALA', 7: 'AWL BANDED COMMON', 8: 'BANDED GOLD', 9: 'BANDED HELICONIAN ORANGE', 10: 'BANDED PEACOCK', 11: 'BARRED FLASHER TWO', 12: 'BECKERS WHITE', 13: 'BIRDWING CAIRNS', 14: 'BLACK HAIRSTREAK', 15: 'BLUE CROW SPOTTED', 16: 'BLUE MORPHO', 17: 'BROWN SIPROETA', 18: 'CABBAGE WHITE', 19: 'CATTLEHEART CELLED GREEN', 20: 'CHECQUERED SKIPPER', 21: 'CHESTNUT', 22: 'CLEOPATRA', 23: 'CLOAK MOURNING', 24: 'CLODIUS PARNASSIAN', 25: 'CLOUDED SULPHUR', 26: 'COMA EASTERN', 27: 'COMMON WOOD-NYMPH', 28: 'COPPER PURPLISH', 29: 'COPPER TAIL', 30: 'CRACKER RED', 31: 'CRECENT', 32: 'CRIMSON PATCH', 33: 'DANAID EGGFLY', 34: 'DAPPLE EASTERN WHITE', 35: 'DOGFACE SOUTHERN', 36: 'EASTERN ELFIN PINE', 37: 'EGGFLY GREAT', 38: 'ELBOWED PIERROT', 39: 'GREAT JAY', 40: 'GREY HAIRSTREAK', 41: 'HAIRSTREAK PURPLE', 42: 'INDRA SWALLOW', 43: 'IPHICLUS SISTER', 44: 'JULIA', 45: 'KITE PAPER', 46: 'LADY PAINTED', 47: 'LARGE MARBLE', 48: 'LEAFWING TROPICAL', 49: 'LONG WING ZEBRA', 50: 'MALACHITE', 51: 'MANGROVE SKIPPER', 52: 'MARK QUESTION', 53: 'MESTRA', 54: 'METALMARK', 55: 'MILBERTS TORTOISESHELL', 56: 'MONARCH', 57: 'OAKLEAF ORANGE', 58: 'ORANGE SLEEPY', 59: 'ORANGE TIP', 60: 'ORCHARD SWALLOW', 61: 'PEACOCK', 62: 'PINE WHITE', 63: 'PIPEVINE SWALLOW', 64: 'POPINJAY', 65: 'POSTMAN RED', 66: 'PURPLE RED SPOTTED', 67: 'QUEEN STRAITED', 68: 'SATYR WOOD', 69: 'SCARCE SWALLOW', 70: 'SILVER SKIPPER SPOT', 71: 'SOOTYWING', 72: 'SWALLOW TAIL YELLOW', 73: 'ULYSES', 74: 'VICEROY'}
inSpanish = False

# Esta parte nos ayuda a debugguear cualquier problema
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Funcion que ocurre cuando se hace /start, describe el modelo y enseña los comandos que se pueden utilizar
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if inSpanish:
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Hola soy NACHI una arquitectura neuronal el cual tiene como uso el reconocimiento de diferentes mariposas, al menos en su primera versión. Puedes utilizarme para identificar a las distintas mariposas, meidante fotos. \n\nPara empezar, solo tienes que mandarme una foto e intentaré predecir qué tipo de mariposa es! \n\nEn caso de querer saber con qué especies de mariposas fui entrenada, puedes usar el comando /species para ver el listado. \n\nSi quieres que el modelo vuelva a hablarte en inglés, utiliza el comando /EN.")
    else:
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Hi I'm NACHI a Neural Architecture who's purpose is to recognize photos of butterflies as of right now in it's first version, please talk to me in case you need help identifying different butterflies! \n\nTo start predicting, just send me a photo of a pretty butterfly! \n\nIn case you want to know what species I have been trained to recognize, use the command /species. \n\nBy the way, if you want this bot to speak in spanish, use the command /ES")

# Función de respuesta por defecto sino se envía imagen
async def respond(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if inSpanish:
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Para empezar, solo tienes que mandarme una foto de una mariposa!")
    else:
        await context.bot.send_message(chat_id=update.effective_chat.id, text="If you want to start predicting, send me a photo of a butterfly!")

# Función que devuelve el listado de clases con las que entrenó el modelo
async def species(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if inSpanish:
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Para tu interés, estas son los diferentes nombres de las especies de las mariposas con las que he sido entrenada: '88 AN', 'ADMIRAL RED', 'ADONIS', 'AFRICAN GIANT SWALLOWTAIL', 'AMERICAN SNOOT', 'APPOLLO', 'ATALA', 'AWL BANDED COMMON', 'BANDED GOLD', 'BANDED HELICONIAN ORANGE', 'BANDED PEACOCK', 'BARRED FLASHER TWO', 'BECKERS WHITE', 'BIRDWING CAIRNS', 'BLACK HAIRSTREAK', 'BLUE CROW SPOTTED', 'BLUE MORPHO', 'BROWN SIPROETA', 'CABBAGE WHITE', 'CATTLEHEART CELLED GREEN', 'CHECQUERED SKIPPER', 'CHESTNUT', 'CLEOPATRA', 'CLOAK MOURNING', 'CLODIUS PARNASSIAN', 'CLOUDED SULPHUR', 'COMA EASTERN', 'COMMON WOOD-NYMPH', 'COPPER PURPLISH', 'COPPER TAIL', 'CRACKER RED', 'CRECENT', 'CRIMSON PATCH', 'DANAID EGGFLY', 'DAPPLE EASTERN WHITE', 'DOGFACE SOUTHERN', 'EASTERN ELFIN PINE', 'EGGFLY GREAT', 'ELBOWED PIERROT', 'GREAT JAY', 'GREY HAIRSTREAK', 'HAIRSTREAK PURPLE', 'INDRA SWALLOW', 'IPHICLUS SISTER', 'JULIA', 'KITE PAPER', 'LADY PAINTED', 'LARGE MARBLE', 'LEAFWING TROPICAL', 'LONG WING ZEBRA', 'MALACHITE', 'MANGROVE SKIPPER', 'MARK QUESTION', 'MESTRA', 'METALMARK', 'MILBERTS TORTOISESHELL', 'MONARCH', 'OAKLEAF ORANGE', 'ORANGE SLEEPY', 'ORANGE TIP', 'ORCHARD SWALLOW', 'PEACOCK', 'PINE WHITE', 'PIPEVINE SWALLOW', 'POPINJAY', 'POSTMAN RED', 'PURPLE RED SPOTTED', 'QUEEN STRAITED', 'SATYR WOOD', 'SCARCE SWALLOW', 'SILVER SKIPPER SPOT', 'SOOTYWING', 'SWALLOW TAIL YELLOW', 'ULYSES', 'VICEROY'.")
    else:
        await context.bot.send_message(chat_id=update.effective_chat.id, text=" \n\nFor your interest, this are the different names of the species of butterflies I have been trained with: '88 AN', 'ADMIRAL RED', 'ADONIS', 'AFRICAN GIANT SWALLOWTAIL', 'AMERICAN SNOOT', 'APPOLLO', 'ATALA', 'AWL BANDED COMMON', 'BANDED GOLD', 'BANDED HELICONIAN ORANGE', 'BANDED PEACOCK', 'BARRED FLASHER TWO', 'BECKERS WHITE', 'BIRDWING CAIRNS', 'BLACK HAIRSTREAK', 'BLUE CROW SPOTTED', 'BLUE MORPHO', 'BROWN SIPROETA', 'CABBAGE WHITE', 'CATTLEHEART CELLED GREEN', 'CHECQUERED SKIPPER', 'CHESTNUT', 'CLEOPATRA', 'CLOAK MOURNING', 'CLODIUS PARNASSIAN', 'CLOUDED SULPHUR', 'COMA EASTERN', 'COMMON WOOD-NYMPH', 'COPPER PURPLISH', 'COPPER TAIL', 'CRACKER RED', 'CRECENT', 'CRIMSON PATCH', 'DANAID EGGFLY', 'DAPPLE EASTERN WHITE', 'DOGFACE SOUTHERN', 'EASTERN ELFIN PINE', 'EGGFLY GREAT', 'ELBOWED PIERROT', 'GREAT JAY', 'GREY HAIRSTREAK', 'HAIRSTREAK PURPLE', 'INDRA SWALLOW', 'IPHICLUS SISTER', 'JULIA', 'KITE PAPER', 'LADY PAINTED', 'LARGE MARBLE', 'LEAFWING TROPICAL', 'LONG WING ZEBRA', 'MALACHITE', 'MANGROVE SKIPPER', 'MARK QUESTION', 'MESTRA', 'METALMARK', 'MILBERTS TORTOISESHELL', 'MONARCH', 'OAKLEAF ORANGE', 'ORANGE SLEEPY', 'ORANGE TIP', 'ORCHARD SWALLOW', 'PEACOCK', 'PINE WHITE', 'PIPEVINE SWALLOW', 'POPINJAY', 'POSTMAN RED', 'PURPLE RED SPOTTED', 'QUEEN STRAITED', 'SATYR WOOD', 'SCARCE SWALLOW', 'SILVER SKIPPER SPOT', 'SOOTYWING', 'SWALLOW TAIL YELLOW', 'ULYSES', 'VICEROY'.")

# Función para ponerlo en español
async def spanish(update: Update, context: ContextTypes.DEFAULT_TYPE):
   global inSpanish
   inSpanish = True
   await context.bot.send_message(chat_id=update.effective_chat.id, text=f"Idioma cambiado correctamente al español.")

# Para ponerlo en inglés
async def english(update: Update, context: ContextTypes.DEFAULT_TYPE):
   global inSpanish
   inSpanish = False
   await context.bot.send_message(chat_id=update.effective_chat.id, text=f"Language changed correctly to english.")

# Función que predice
async def predict(update: Update, context: ContextTypes.DEFAULT_TYPE):

    # Obtener la foto de mayor resolución
    photo = update.message.photo[-1]
    file = await context.bot.get_file(photo.file_id)
    file_path = f"downloads/{photo.file_id}.jpg"
    
    if not os.path.exists('downloads'):
        os.makedirs('downloads')

    # Descargar la foto
    await file.download_to_drive(file_path)
    
    # Preprocesamiento de la imagen
    img = image.load_img(file_path, target_size=(180, 180))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) 
    
    # Realiza la predicción
    predictions = model.predict(img_array)
    
    # Se saca las 3 clases con mayor índice de confianza
    top_indices = np.argsort(predictions[0])[-3:][::-1]
    
    # Se crea la respuesta, donde se pone la clase, su índice y la confianza
    response_message = ""
    for i in top_indices:
        class_name = dict_classes.get(i)
        confidence = predictions[0][i]
        if inSpanish:
            response_message += f"Predicción clase {i}: {class_name}, Confianza: {confidence:.3f}\n"
        else:
            response_message += f"Prediction class {i}: {class_name}, Confidence: {confidence:.3f}\n"
    
    # Eliminar el archivo descargado
    os.remove(file_path)
    
    # Enviar la respuesta al usuario
    await context.bot.send_message(chat_id=update.effective_chat.id, text=response_message)



if __name__ == '__main__':
    application = ApplicationBuilder().token(os.getenv("API_KEY")).build()
    
    start_handler = CommandHandler('start', start)
    application.add_handler(start_handler)

    species_handler = CommandHandler('species', species)
    application.add_handler(species_handler)

    echo_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), respond)
    application.add_handler(echo_handler)
    
    predict_handler = MessageHandler(filters.PHOTO, predict)
    application.add_handler(predict_handler)

    spanish_handler = CommandHandler('ES', spanish)
    application.add_handler(spanish_handler)

    english_handler = CommandHandler('EN', english)
    application.add_handler(english_handler)

    application.run_polling()