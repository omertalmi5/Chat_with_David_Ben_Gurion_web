import streamlit as st
import asyncio
import os
from dotenv import load_dotenv
import groq
from functools import lru_cache
import fitz  # PyMuPDF for PDF extraction
from sentence_transformers import SentenceTransformer, util  # For semantic search
import re

# Load environment variables
load_dotenv()

# Groq API setup
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = groq.Groq(api_key=GROQ_API_KEY)
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")
QUOTES_FILE_PATH = "./quotes_document_edited.pdf"

# Here is the long Python string as requested:

style_quotes = """
לשם ראיית המאבק האידיאי והבנת ערכו בהתרוצצות ההיסטורית, אין הכרח לברר ולהכריע בוויכוח
הפילוסופי אם מאבק אידיאי נובע מתוך ניגודים כלכליים, חברתיים ומדיניים או שהוא מחולל אותם,
או אם הניגודים הכלכליים והרעיוניים כרוכים זה בזה ואין להפריד בין הדבקים. אין כל ערך מעשי
ו"נפקא מינה" לוויכוח מופשט זה, כשם שאין ערך מעשי לבירור אם התרנגולת קדמה לביצה או
שהביצה קדמה לתרנגולת. ברור שאי-אפשר לזו בלא זו. ואין כל אפשרות לגדל תרנגולות בלי דגירת
ביצים ,ואין כל דרך להשיג ביצים אלא על-ידי גידול תרנגולות. וראינו בהיסטוריה אידיאות ששינו
משטרים – פוליטיים וכלכליים, וראינו משטרים שחידשו אידיאות והשליטו אותן. אנשים נלחמים על
דעותיהם לא פחות מאשר על שלטונם ורכושם, ומזמן שהאדם עמד על דעתו – לא חדל המאבק
האידיאי, ובתולדות עמנו הוא תופס מקום יותר נרחב אולי מאשר בתולדות כל עם ועם. וכמעט שלא
היה אף מאבק אחד בתולדותינו, מדיני או צבאי, שלא היה כרוך במאבק אידיאי.
*
אנו עומדים עכשיו בסכסוך לא רק עם שכנינו הערבים, אלא במידה ידועה עם רוב העולם האנושי,
כפי שהוא מאורגן באומות המאוחדות, בגלל ירושלים. ורק עיוור לא יראה שמקורות הסכסוך הזה
אינם אך ורק פוליטיים, כלכליים או צבאיים – אלא גם אידיאיים.
כשהסורים, העיראקים והמצרים תומכים בהתלהבות, כביכול, בבינאום ירושלים הרי
נימוקיהם ברורים: מוטב שמסגד עומר ָ ימצא תחת שלטון נוצרי מאשר ימצא חלק גדול של ירושלים
תחת שלטון יהודי. אבל קשה להסביר בנימוקים פוליטיים בלבד עמדתן של כמה אומות באמריקה
הדרומית, אשר בדרך כלל עמדו לימיננו באו"ם במאבקנו המדיני, ובשאלת ירושלים הפכו נגדנו. אין
להסביר בנימוקים מדיניים עמדת צרפת, אשר היה לה ענין מדיני וצבאי רב לעזור לנו, וגם עזרה לנו
לא מעט, לא רק בעצרת האו"ם, אלא בדברים הרבה יותר ממשיים ויעילים. והוא הדין
בצ'כוסלובקיה. ואף-על-פי-כן יצאו אומות אלה נגדנו בשאלת ירושלים. אין להתעלם מהעובדה שיש
גם מאבק אידיאי בעולם.
בשאלת ירושלים ראינו צירוף משונה ותמוה מאוד. מצד אחד עמד, אם לא כל העולם הנוצרי,
הרי הגוש האוניברסלי הגדול ביותר בעולם הנוצרי, הגוש הקתולי. מהצד השני עמד הגוש המוסלמי.
מהצד השלישי – הגוש הקומוניסטי.
אין ספק שלכל אחד מהגושים הללו היו נימוקים משלו. אבל אין ספק שהיה גם צד שווה, אם
כי לא משותף, לשלושת הגושים האלה. מה שמאחד כל גוש הוא לא רק אינטרס מדיני, אלא גם
אידיאה. יש אידיאה באיסלם, יש אידיאה בקתוליות, יש אידיאה בקומוניזם. ושוב לא מענין לגבי הבנת
בעייתנו, אם האידיאה קובעת המדיניות או המדיניות קובעת האידיאה, שתיהן יחד נובעות ממקור
אחד. התעלמות מהאידיאה היא התעלמות מאחד הגורמים והגילויים המרכזיים בהיסטוריה
האנושית.
"""

life_summary = """דוד גרין נולד בעיירה פלונסק שבפולין, בשנת .1886 בגיל 14 ייסד עם כמה נערים בני עירו אגודה ציונית בשם
"עזרא", שחבריה התחייבו לדבר עברית. בשנת 1906 עלה לארץ ליפו ולאחר מכן עבר לסג'רה שבצפון, עם
חברי העלייה השנייה. במהלך שנות קליטתו בארץ שינה את שמו לבן-גוריון. בשנת 1917 שהה בארצות הברית
ופגש שם את פולה מונבז. פולה, כך קרא לה דוד, נולדה במינסק שבבלרוס בשם פאולינה. כשהייתה בת 12
עברה משפחתה לניו יורק. כשנפגשו פולה ודוד בניו יורק הם התאהבו מיד. באותם הימים עבדה פולה כאחות
בחדר ניתוח. לדוד לא היו הרבה מכרים באותה העת, והוא היה בודד. לכן ביקש ממנה שתבוא איתו לספרייה
הציבורית לעזור לו להעתיק את מה שמספרים על העתיד של ארץ ישראל... דוד ביקש את ידה של פולה, אך
לפני החתונה המיוחלת הציב בפני פולה תנאי אחד: להקים בית ומשפחה בארץ ישראל. "מן היום שבו הכרתי
את בן-גוריון, ידעתי כי הוא אדם גדול", אמרה פעם פולה.
השניים התחתנו בשנת 1917 בניו יורק. אחרי שהטקס הסתיים נסעה פולה למשמרת בבית החולים, ודוד נסע
לפגישת עבודה. הם הקימו בית בישראל ונולדו להם שלושה ילדים.
פועלו של דוד בן-גוריון בארץ היה רב ומגוון: הוא נמנה עם מקימי מפלגת אחדות העבודה,
הוא כיהן כמזכיר הכללי של הסתדרות העובדים בארץ ישראל, כיושב ראש הנהלת הסוכנות היהודית,
כראש הממשלה הראשון של מדינת ישראל וכשר הביטחון. בשנת 1953 התפטר, ועלה לשדה בוקר בנגב."""

# Set page config at the very beginning
st.set_page_config(layout="wide", page_title="Chat with David Ben Gurion", page_icon="🌟")

# Add RTL support via CSS and set background and text colors for desert style
desert_style = """
    <style>
    body {
        direction: rtl;
        color: #8B4513; /* Warm brown text color */
    }
    h1 {
        text-align: center;
        color: #ffffff; /* Warm desert orange-brown for the title */
        font-family: 'Arial', sans-serif;
    }
    .stTextInput > div > div > input {
        direction: rtl;
        background-color: #FBEEC1; /* Light sand-colored input field */
        border: 2px solid #C19A6B; /* Warm brown border */
        # color: #8B4513; /* Brown text inside input */
    }
    .stChatMessage .stMarkdown {
        text-align: right;
        # color: #8B4513; /* Ensure chat text is warm brown */
    }
    </style>
"""

# Apply the desert style
st.markdown(desert_style, unsafe_allow_html=True)

# Display the title in the styled way (warm desert orange-brown color)
st.markdown("<h1>סליחה על השאלה עם דוד בן גוריון</h1>", unsafe_allow_html=True)

# Display the image after the title
col1, col2, col3 = st.columns([1, 3, 1])  # Center the image
with col2:
    st.image("main_image.PNG", width = 30, use_column_width=True)

# Load the pre-trained sentence-transformers model for semantic search
@st.cache_resource
def load_model():
    model = SentenceTransformer('intfloat/multilingual-e5-large')  # Lightweight transformer model for semantic search
    return model


model = load_model()


# Function to extract and chunk quotes from the PDF document
@st.cache_data
def load_quotes_from_pdf(pdf_path, chunk_size=200):
    doc = fitz.open(pdf_path)
    text = ""

    # Extract text from each page
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text += page.get_text("text")

    split_signs = r'[.!?\n\(\)]'

    # List of unwanted strings
    unwanted_strings = ["עמ'", "עמ׳", "חזון ודרך", "עמ׳"]  # Add more unwanted strings here
    string_number_pattern = r'\D+\d+'
    # Sample code with filtering conditions
    filtered_sentences = [
        re.sub(r'\d+', '', sentence)
        for sentence in re.split(split_signs, text)  # Split by multiple signs including new lines and parentheses
        if not any(unwanted in sentence for unwanted in unwanted_strings)
           and not re.search(string_number_pattern, sentence)
    ]

    # Chunking the text by combining sentences into manageable chunks (e.g., 200-300 words per chunk)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in filtered_sentences:
        current_chunk.append(sentence)
        current_length += len(sentence.split())

        # If the current chunk reaches the desired size, save it and start a new one
        if current_length >= chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0

    # Add any remaining text as a chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    print(f"Total chunks: {len(chunks)}")
    return chunks

# Load chunks from the attached PDF
pdf_quotes = load_quotes_from_pdf(QUOTES_FILE_PATH)

with open('wiki.txt', 'r', encoding='utf-8') as file:
    wiki_quotes = file.read().splitlines()

QUOTES = wiki_quotes + pdf_quotes
# Precompute embeddings for all the chunks
@st.cache_data
def compute_quote_embeddings(quotes):
    return model.encode(quotes, convert_to_tensor=True)

QUOTE_EMBEDDINGS = compute_quote_embeddings(QUOTES)

# Function to retrieve relevant chunks based on semantic similarity
def retrieve_relevant_quotes_semantically(query, top_k=3):
    query_embedding = model.encode(query, convert_to_tensor=True)
    # Compute cosine similarity between the query and all the quote embeddings
    cos_scores = util.pytorch_cos_sim(query_embedding, QUOTE_EMBEDDINGS)[0]

    # Convert cos_scores tensor to a list of tuples (score, index)
    scores_and_indices = [(score.item(), idx) for idx, score in enumerate(cos_scores)]

    # Sort by cosine similarity score in descending order and return top_k results
    top_results = sorted(scores_and_indices, key=lambda x: x[0], reverse=True)[:top_k]

    # Retrieve the top k chunks based on sorted scores
    relevant_quotes = [QUOTES[idx] for _, idx in top_results]
    print(len(relevant_quotes))

    if not relevant_quotes:
        return ["No relevant quotes found."]

    return relevant_quotes



@lru_cache(maxsize=100)
def ask_groq(question):
    # Retrieve relevant quotes from the PDF using semantic search
    relevant_quotes = retrieve_relevant_quotes_semantically(question, 3)

    # Join relevant quotes into a single string for the system prompt
    quotes_context = "\n".join(relevant_quotes)
    print(quotes_context)
    # Include the retrieved quotes in the system prompt to guide the response
    system_prompt = (f"You are David Ben Gurion, the first Prime Minister of Israel, and should answer as if you are him. Use first person, answer only in Hebrew, keep correct Henrew, and adopt the jargon of the 1960s. If unsure about a response say that you don't rememebr and need to check it. Talks about only child friendly topics."
                     f" {style_quotes}, here summary of your life: {life_summary}")

    context_prompt = f"If it relevant base your answer on the following quotes of you to guide your response:\n\n{quotes_context}\n\n"

    try:
        response = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "assistant", "content": context_prompt},
                {"role": "user", "content": question},
            ],
            model=GROQ_MODEL,
            temperature=0.8,
            max_tokens=1024,
            stream=False,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return "סליחה אך אינני יכול לענות על שאלה זו, נסה שאלה אחרת"


def create_chatbot():
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []

    # Display previous chat messages
    for message in st.session_state.get('messages', []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input for the user's message
    prompt = st.chat_input("שאל אותי כל שאלה")

    if prompt:
        # Display the user input
        st.chat_message("user").markdown(prompt)
        st.session_state['messages'].append({"role": "user", "content": prompt})

        # Generate response from David Ben Gurion using retrieved quotes
        with st.chat_message("assistant"):
            with st.spinner('אני חושב...'):
                response = ask_groq(prompt)
            st.markdown(response)
        st.session_state['messages'].append({"role": "assistant", "content": response})


async def main():
    create_chatbot()


if __name__ == "__main__":
    asyncio.run(main())
