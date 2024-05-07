import streamlit as st
import random
import numpy as np
from sklearn.metrics import cohen_kappa_score
from gensim.models import Word2Vec
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from transformers import BertTokenizer, BertModel
import transformers as ppb
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

#nltk.download('punkt')
#nltk.download('stopwords')

def split_in_sets(data):
    essay_sets = []
    min_scores = []
    max_scores = []

    for s in range(1, 9):
        essay_set = data[data["essay_set"] == s]
        essay_set.dropna(axis=1, inplace=True)
        n, d = essay_set.shape
        set_scores = essay_set["domain1_score"]
        print("Set", s, ": Essays = ", n, "\t Attributes = ", d)
        min_scores.append(set_scores.min())
        max_scores.append(set_scores.max())
        essay_sets.append(essay_set)

    return essay_sets, min_scores, max_scores

def stop_words():
    # Daftar kata kunci untuk entitas bernama dalam bahasa Indonesia
    nama_tempat = ['@LOCATION' + str(i) for i in range(100)]
    organisasi = ['@ORGANIZATION' + str(i) for i in range(100)]
    orang = ['@PERSON' + str(i) for i in range(100)]
    tanggal = ['@DATE' + str(i) for i in range(100)]
    waktu = ['@TIME' + str(i) for i in range(100)]
    uang = ['@MONEY' + str(i) for i in range(100)]

    # Menggabungkan semua daftar kata kunci menjadi satu daftar
    stop_words_id = nama_tempat + organisasi + orang + tanggal + waktu + uang

    return stop_words_id

def remove_special_chars(text):
    # Menghapus karakter tab, baris baru, back slice, dan karakter non-ASCII
    text = re.sub(r'[\t\n\r\x0c\x0b]', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)

    # Mengubah teks menjadi huruf kecil
    text = text.lower()

    # Menghapus tanda baca
    text = re.sub(r'[^\w\s]', '', text)

    # Menghapus mention, link, hashtag, dan URL yang tidak lengkap
    text = re.sub(r'(@\w+)|((https?://|www\.)?\w+\.\w+)', '', text)

    return text

def tokenize(text):
    return word_tokenize(text)

def stem_text(text):
    stemmer = PorterStemmer()
    tokens = tokenize(text)
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)

def remove_stop_words(text):
    stop_words = set(stopwords.words('indonesian'))
    tokens = tokenize(text)
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
    return ' '.join(filtered_tokens)

def preprocess_text(text):
    text = remove_special_chars(text)
    stemmed_text = stem_text(text)
    filtered_text = remove_stop_words(stemmed_text)
    return filtered_text

def calculate_avg_sentence_length(essay):
    # Tokenisasi kalimat
    sentences = nltk.sent_tokenize(essay)

    # Menghitung panjang setiap kalimat
    sentence_lengths = [len(re.findall(r'\w+', sentence)) for sentence in sentences]

    # Menghitung rata-rata panjang kalimat
    avg_sentence_length = sum(sentence_lengths) / len(sentence_lengths)

    return avg_sentence_length

def calculate_sentence_coherence(essay):
    # Tokenisasi kalimat
    sentences = nltk.sent_tokenize(essay)

    # Menghitung koherensi antar kalimat
    coherence_scores = []
    for i in range(len(sentences) - 1):
        sent1 = sentences[i]
        sent2 = sentences[i + 1]

        # Tokenisasi kata dan penghapusan stopwords
        words1 = [word.lower() for word in re.findall(r'\w+', sent1) if word.lower() not in stopwords.words('english')]
        words2 = [word.lower() for word in re.findall(r'\w+', sent2) if word.lower() not in stopwords.words('english')]

        # Menghitung kemiripan antar kalimat menggunakan overlapping kata
        overlap = len(set(words1).intersection(set(words2)))
        coherence_score = overlap / max(len(words1), len(words2))
        coherence_scores.append(coherence_score)

    # Menghitung rata-rata koherensi kalimat
    avg_coherence = sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0

    return avg_coherence * 100

def get_bert_score(essay, model, tokenizer, cuda):
    encoding = tokenizer(essay, truncation=True, padding=True, max_length=512, return_tensors='pt')
    input_ids, attention_mask = encoding['input_ids'].to(cuda), encoding['attention_mask'].to(cuda)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        score = outputs.logits.cpu().numpy().flatten()[0]

        # Normalisasi skor ke rentang 0 hingga 100
        min_score = -10  # Asumsi skor minimum adalah -10
        max_score = 10   # Asumsi skor maksimum adalah 10
        normalized_score = (score - min_score) / (max_score - min_score) * 100
        normalized_score = max(normalized_score, 0)  # Memastikan skor tidak negatif
        normalized_score = int(round(normalized_score))

    return normalized_score

dataset_path='D:\\Kuliah\\BISMILLAH TAMAT TEPAT WAKTU SEBELUM BULAN 5\\WebsiteEssay\\preprocess_dataset (2).tsv'

data = pd.read_csv(dataset_path, delimiter='\t')
X = data['essay']
y = data['domain1_score']

from gensim.models import Word2Vec

# Membagi data menjadi train dan test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Membuat WordVectors
word2vec_model = Word2Vec.load('D:\\Kuliah\\BISMILLAH TAMAT TEPAT WAKTU SEBELUM BULAN 5\\WebsiteEssay\\word2vec_model_fold_1 (3).bin')
vocab = set(word2vec_model.wv.key_to_index.keys())

# Tokenizer
tokenizer = lambda text: [word for word in text.split() if word in vocab]

# Konversi teks menjadi tensor
def text_to_tensor(text, tokenizer, word2vec_model):
    tokens = tokenizer(text)
    word_vectors = [word2vec_model.wv[token] for token in tokens]
    tensor = torch.tensor(word_vectors)
    return tensor

from torch.nn.utils.rnn import pad_sequence

# Membuat tensor untuk data train dan test
X_train_tensors = [text_to_tensor(text, tokenizer, word2vec_model) for text in X_train]
X_train_tensor = pad_sequence(X_train_tensors, batch_first=True)
X_test_tensors = [text_to_tensor(text, tokenizer, word2vec_model) for text in X_test]
X_test_tensor = pad_sequence(X_test_tensors, batch_first=True)

y_train_tensor = torch.tensor(y_train.values.reshape(-1), dtype=torch.float)
y_test_tensor = torch.tensor(y_test.values.reshape(-1), dtype=torch.float)

# Membuat dataset dan dataloader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Model Bi-GRU untuk umpan balik
class BiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(BiGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, 1, self.hidden_size)
        out, _ = self.gru(x, h0)
        out = out[:, -1, :]  # Mengambil output dari hidden state terakhir
        out = self.fc(out)
        return out

# Inisialisasi model Bi-GRU
input_size = word2vec_model.vector_size
hidden_size = 128
output_size = len(set(y_train_tensor))  # Jumlah kategori umpan balik
model = BiGRU(input_size, hidden_size, output_size)

# Memuat model Bi-GRU yang telah dilatih
cuda = torch.device('cuda')
bigru_model = BiGRU(input_size, hidden_size, output_size)
bigru_model.load_state_dict(torch.load('C:\\Users\\Faradhila\\Downloads\\bigru_model.pth'))
bigru_model.to(cuda)
bigru_model.eval()

def generate_feedback(final_score):
    if final_score >= 90:
        feedback = "Esai Anda sangat baik dan memenuhi semua kriteria yang diharapkan. Ide utama tersampaikan dengan jelas, struktur kalimat dan koherensi sangat baik."
    elif final_score >= 80:
        feedback = "Esai Anda cukup baik. Ide utama tersampaikan dengan cukup jelas, namun masih ada ruang untuk peningkatan dalam struktur kalimat dan koherensi."
    elif final_score >= 70:
        feedback = "Esai Anda sudah cukup baik, namun masih perlu perbaikan dalam penyampaian ide utama, struktur kalimat, dan koherensi."
    elif final_score >= 60:
        feedback = "Esai Anda masih membutuhkan perbaikan signifikan dalam penyampaian ide utama, struktur kalimat, dan koherensi."
    else:
        feedback = "Esai Anda masih kurang memenuhi kriteria yang diharapkan. Anda perlu meningkatkan penyampaian ide utama, struktur kalimat, dan koherensi."

    return feedback

def predict_essay_score(essay, prompt, reference_essays, model, tokenizer, cuda, word2vec_model, bigru_model):
    # Mendapatkan skor dari model BERT
    essay = essay
    # Convert user essay to tensors
    input_ids = torch.tensor(tokenizer(essay)['input_ids']).unsqueeze(0)
    attention_mask = torch.tensor(tokenizer(essay)['attention_mask']).unsqueeze(0)

    # Move input tensors to the same device as the model
    if cuda:
        input_ids = input_ids.to(cuda)
        attention_mask = attention_mask.to(cuda)

    # Calculate BERT score
    with torch.no_grad():
        model.to(cuda)  # Move the model to the same device as the input tensors
        outputs = model(input_ids, attention_mask=attention_mask)
        bert_score = bert_score = get_bert_score(essay, model, tokenizer, cuda)

    # Mendapatkan skor text similarity
    text_similarity = get_text_similarity(essay, prompt, reference_essays, word2vec_model)

    # Menghitung komponen rubrik skor
    avg_sentence_length = calculate_avg_sentence_length(essay)
    sentence_coherence = calculate_sentence_coherence(essay)
    main_idea_score = check_main_idea(essay, prompt, reference_essays)

    # Menghitung rubrik skor
    rubric_score = compute_rubric_score(avg_sentence_length, sentence_coherence, main_idea_score)


    # Mencetak nilai bert_score, text_similarity, dan rubric_score
    print(f"BERT Score: {bert_score}")
    print(f"Text Similarity: {text_similarity}")
    print(f"Rubric Score: {rubric_score}")

    # Menghitung skor akhir
    combined_score = bert_score + text_similarity + rubric_score
    print(f"Combined Score (bert_score + text_similarity + rubric_score): {combined_score}")

    # Memberikan umpan balik berdasarkan skor akhir
    final_score = combined_score / 3
    feedback = generate_feedback(final_score)



    return final_score, feedback

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


import re
from collections import Counter

def check_main_idea(essay, prompt, reference_essays):
    # Mengambil kata kunci dari prompt
    prompt_keywords = re.findall(r'\w+', prompt.lower())

    # Mengambil kata kunci dari referensi esai
    reference_keywords = []
    for ref_essay in reference_essays:
        reference_keywords.extend(re.findall(r'\w+', ref_essay.lower()))

    # Menggabungkan kata kunci dari prompt dan referensi esai
    all_keywords = prompt_keywords + reference_keywords

    # Menghitung frekuensi kemunculan kata kunci dalam esai
    essay_words = re.findall(r'\w+', essay.lower())
    keyword_counts = Counter(essay_words)

    # Menghitung skor main idea berdasarkan kemunculan kata kunci
    total_keyword_count = sum(keyword_counts[keyword] for keyword in all_keywords)
    main_idea_score = float(total_keyword_count) / len(all_keywords)

    return main_idea_score

def get_text_similarity(essay, prompt, reference_essays, word2vec_model):
    # Membuat representasi vektor dari essay
    essay_words = essay.split()
    essay_vectors = []
    for word in essay_words:
        try:
            essay_vectors.append(word2vec_model.wv[word])
        except KeyError:
            # Jika kata tidak ditemukan di dalam model word2vec, abaikan
            pass

    if len(essay_vectors) == 0:
        # Jika essay tidak memiliki kata apa pun, return skor 2
        print("Essay kosong, mengembalikan skor 2")
        return 2

    essay_vector = np.mean(essay_vectors, axis=0)
    print(f"Essay vector: {essay_vector}")

    # Membuat representasi vektor dari kumpulan essay referensi
    reference_vectors = []
    for ref_essay in reference_essays:
        ref_words = ref_essay.split()
        ref_vectors = []
        for word in ref_words:
            try:
                ref_vectors.append(word2vec_model.wv[word])
            except KeyError:
                pass
        if len(ref_vectors) == 0:
            # Jika essay referensi tidak memiliki kata apa pun, return skor 2
            print("Essay referensi kosong, mengembalikan skor 2")
            return 2
        ref_vector = np.mean(ref_vectors, axis=0)
        reference_vectors.append(ref_vector)

    # Menghitung similarity cosine antara vektor essay dengan setiap vektor referensi
    similarities = []
    for ref_vector in reference_vectors:
        similarity = cosine_similarity(essay_vector.reshape(1, -1), ref_vector.reshape(1, -1))
        similarities.append(similarity[0][0])
        print(f"Cosine similarity dengan essay referensi ke-{np.where(np.all(reference_vectors == ref_vector, axis=1))[0]}: {similarity[0][0]}")


    # Mengembalikan skor
    if len(similarities) == 0:
        # Jika tidak ada essay referensi yang memiliki kata, return skor 3
        print("Tidak ada essay referensi yang memiliki kata, mengembalikan skor 3")
        return 3
    else:
        # Jika essay dan essay referensi memiliki kata, return nilai rata-rata similarity
        skor = (np.mean(similarities)) * 100
        print(f"Skor text similarity: {skor}")
        return skor


def compute_rubric_score(avg_sentence_length, sentence_coherence, main_idea_score):
    # Normalisasi setiap komponen ke rentang 0 hingga 1
    normalized_avg_sentence_length = min(avg_sentence_length / 20, 1)
    normalized_sentence_coherence = min(sentence_coherence / 100, 1)
    normalized_main_idea_score = min(main_idea_score, 1)

    # Menghitung skor akhir dengan bobot yang sesuai
    score = (normalized_avg_sentence_length * 0.4) + (normalized_sentence_coherence * 0.2) + (normalized_main_idea_score * 0.4)

    # Membatasi skor ke rentang 0 hingga 100
    rubric_score = int(max(min(score, 1), 0) * 100)

    return rubric_score

# Memuat model Word2Vec
word2vec_model = Word2Vec.load('C:\\Users\\Faradhila\\Downloads\\word2vec_model_fold_1 (3).bin')


# Inisialisasi model, tokenizer, dan word2vec

#bert_model_path='bert-base-uncased'
#cuda = torch.device('cuda')
#tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
#model = AutoModelForSequenceClassification.from_pretrained(bert_model_path, num_labels=1)
#model.load_state_dict(torch.load('C:\\Users\\Faradhila\\Downloads\\bert_model_kedua.pt', map_location=torch.device('cpu')))
#model.to(torch.device('cuda'))
#word2vec_model = Word2Vec.load("C:\\Users\\Faradhila\\Downloads\\word2vec_model_fold_1_kedua.bin")

# Load pre-trained models and tokenizers
bert_model_path='bert-base-uncased'
cuda = torch.device('cuda')
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
word2vec_model = Word2Vec.load('D:\\Kuliah\\BISMILLAH TAMAT TEPAT WAKTU SEBELUM BULAN 5\\WebsiteEssay\word2vec_model_fold_1 (3).bin')
model = AutoModelForSequenceClassification.from_pretrained(bert_model_path, num_labels=1)
model.load_state_dict(torch.load('D:\\Kuliah\\BISMILLAH TAMAT TEPAT WAKTU SEBELUM BULAN 5\\WebsiteEssay\\bertmodel30epoch.pt', map_location=torch.device('cpu')))
bigru_model = BiGRU(input_size, hidden_size, output_size)
bigru_model.load_state_dict(torch.load("C:\\Users\\Faradhila\\Downloads\\bigru_model.pth"))

# Mendefinisikan prompt dan referensi esai
prompts = [
    {
        "prompt": "Tulis sebuah esai tentang pentingnya menjaga lingkungan.",
        "reference_essays": [
            "Lingkungan yang bersih dan sehat sangat penting bagi kehidupan manusia. Dengan menjaga lingkungan, kita dapat mencegah berbagai masalah seperti pencemaran udara, tanah, dan air, yang dapat berdampak buruk pada kesehatan kita.",
            "Menjaga lingkungan tidak hanya penting untuk generasi sekarang, tetapi juga untuk generasi yang akan datang. Dengan melestarikan alam, kita memastikan bahwa anak-anak kita dan cucu-cucu kita dapat menikmati sumber daya alam yang masih tersedia."
        ]
    },

    {
        "prompt": "Machine Learning merupakan hal yang menjadi pembelajaran yang disukai mahasiswa, bisakah kamu menjelaskan kenapa machine learning menjadi kesukaan mahasiswa.",
        "reference_essays": [
            "Machine learning adalah cabang dari kecerdasan buatan di mana komputer belajar dari data yang diberikan untuk meningkatkan kinerjanya dalam menyelesaikan tugas tertentu tanpa perlu pemrograman eksplisit. Ini berbeda dari pendekatan konvensional di mana algoritma diprogram secara langsung untuk mengekstraksi informasi dari data. Proses utama dalam machine learning melibatkan persiapan data, pemilihan model, pelatihan model, evaluasi, dan iterasi. Persiapan data adalah langkah awal di mana data dikumpulkan, dibersihkan, dan diubah menjadi format yang sesuai untuk analisis. Selanjutnya, model yang sesuai dipilih berdasarkan jenis masalah yang ingin dipecahkan, seperti klasifikasi atau regresi. Setelah model dipilih, dilakukan pelatihan dengan menggunakan data yang telah disiapkan sebelumnya. Proses pelatihan ini mengharuskan model untuk menyesuaikan parameter internalnya agar dapat mengenali pola yang ada dalam data. Ketika pelatihan selesai, model dievaluasi menggunakan data yang belum pernah dilihat sebelumnya untuk menguji kemampuannya dalam melakukan prediksi yang akurat. Evaluasi ini memastikan bahwa model dapat menangani data baru dengan baik, bukan hanya mengingat pola yang sudah dikenal. Selanjutnya, model ditingkatkan melalui iterasi, di mana umpan balik dari evaluasi digunakan untuk memperbaiki model. Inti dari machine learning adalah kemampuan model untuk mengenali pola dalam data dan membuat prediksi yang berguna. Dengan kemampuan ini, machine learning digunakan dalam berbagai aplikasi, termasuk pengenalan wajah, analisis sentimen, dan rekomendasi produk. Seiring dengan perkembangan teknologi dan pengumpulan data yang semakin besar, machine learning menjadi semakin penting dalam menganalisis dan memahami dunia di sekitar kita.",
            "Machine learning adalah paradigma dalam kecerdasan buatan di mana komputer belajar dari data untuk meningkatkan kinerjanya dalam menyelesaikan tugas tertentu. Berbeda dengan pendekatan konvensional yang mengharuskan pemrograman langsung, machine learning memungkinkan komputer untuk mengekstrak pola dari data tanpa instruksi eksplisit. Proses utamanya meliputi persiapan data, pemilihan model yang cocok, pelatihan model dengan data yang disiapkan, evaluasi kinerja model menggunakan data yang belum pernah dilihat sebelumnya, dan iterasi untuk meningkatkan kualitas model.Persiapan data adalah tahap awal di mana data dikumpulkan, disaring, dan disesuaikan agar cocok untuk analisis. Kemudian, model yang paling sesuai dipilih berdasarkan jenis masalah yang ingin dipecahkan, seperti  klasifikasi atau regresi. Setelah itu, model dilatih menggunakan data yang telah dipersiapkan sebelumnya. Proses pelatihan ini memungkinkan model untuk menyesuaikan diri dengan pola dalam data dengan menyesuaikan parameter internalnya. Setelah pelatihan selesai, model dievaluasi menggunakan data pengujian yang tidak dikenal sebelumnya untuk memastikan kemampuannya dalam membuat prediksi yang akurat. Evaluasi ini penting untuk memastikan bahwa model dapat diterapkan pada data baru dengan performa yang baik. Selanjutnya, model dapat ditingkatkan melalui iterasi, di mana umpan balik dari evaluasi digunakan untuk memperbaiki model. Inti dari machine learning adalah kemampuan model untuk mengenali pola dalam data dan membuat prediksi yang bermanfaat. Dengan kemampuan ini, machine learning diterapkan dalam berbagai bidang, termasuk pengenalan wajah, analisis sentimen, dan rekomendasi produk. Seiring dengan perkembangan teknologi dan pertumbuhan besar dalam kuantitas data yang tersedia, machine learning menjadi semakin penting dalam menganalisis dan memahami dunia di sekitar kita."
        ]
    },

    {
        "prompt": "Jika CNN merupakan sebuah metode machine learning yang dikhusus kan untuk data gambar, menurut kamu apakah CNN bisa digunakan dalam data teks ataupun suara, jelaskan.",
        "reference_essays": [
            "CNN digunakan untuk memproses gambar, namun bisa diadaptasi untuk teks atau suara. Untuk teks, CNN menganalisis urutan kata atau karakter untuk pengenalan pola seperti kata kunci atau frase, berguna untuk klasifikasi teks atau analisis sentimen. Untuk suara, CNN dapat mengidentifikasi suara dengan belajar fitur-fitur audio yang relevan. Meskipun CNN awalnya untuk gambar, perlu penyesuaian untuk teks atau suara. Intinya, CNN mengekstraksi fitur penting dari data spasial untuk pemrosesan data yang kompleks.",
            "Meskipun CNN dikembangkan untuk data gambar, konsep dasarnya bisa diadaptasi untuk teks atau suara. Dalam teks, CNN dapat menganalisis urutan kata untuk pengenalan pola, sementara dalam suara, CNN dapat mengidentifikasi fitur-fitur audio penting. Meskipun memerlukan penyesuaian, inti dari CNN tetap sama: mengekstraksi fitur penting untuk pemrosesan data yang kompleks."
        ]
    },

    {
        "prompt": "Tuliskan esai yang menjelaskan peran penting deep learning dalam perkembangan teknologi modern. Jelaskan bagaimana deep learning telah mengubah paradigma di berbagai bidang, seperti pengenalan gambar, bahasa alami, dan pengolahan suara. Diskusikan juga tantangan dan potensi masa depan dalam pengembangan deep learning.",
        "reference_essays": [
            "Deep learning, terutama melalui Convolutional Neural Networks (CNN) untuk pengenalan gambar dan model seperti Transformer untuk pemrosesan bahasa alami, telah mengubah teknologi modern. CNN memungkinkan sistem mengenali objek dengan akurasi tinggi, diterapkan dalam kendaraan otonom dan pengenalan wajah. Sementara Transformer meningkatkan terjemahan otomatis dan chatbot yang lebih responsif. Namun, tantangan seperti interpretasi model dan ketersediaan data berkualitas tetap menjadi fokus pengembangan deep learning.",
            "Peran deep learning dalam era teknologi modern tidak dapat diabaikan. Dalam pengolahan suara, model seperti Recurrent Neural Networks (RNN) telah mengubah cara kita berinteraksi dengan perangkat, memungkinkan pengenalan suara yang lebih baik dan pengembangan asisten virtual yang lebih cerdas. Selain itu, dalam bidang kesehatan, deep learning telah memungkinkan deteksi penyakit dengan cepat dan akurat melalui analisis citra medis. Namun, penting untuk mengakui tantangan seperti keamanan dan privasi data dalam pemanfaatan teknologi deep learning ini."
        ]
    },

    {
        "prompt": "Tuliskan esai yang menjelaskan kegunaan dan signifikansi penggunaan BERT (Bidirectional Encoder Representations from Transformers) dalam mengolah data teks. Jelaskan bagaimana BERT telah meningkatkan pemahaman komputer terhadap konteks dan kompleksitas teks, serta dampaknya dalam berbagai aplikasi, seperti pemrosesan bahasa alami, analisis sentimen, dan pemahaman pertanyaan-jawaban.",
        "reference_essays": [
            "BERT memainkan peran penting dalam pengolahan data teks modern. Dengan kemampuannya memahami konteks dan hubungan antar kata, BERT meningkatkan pemrosesan bahasa alami, meningkatkan analisis sentimen, dan memungkinkan pemahaman pertanyaan-jawaban yang lebih baik. Ini menghasilkan analisis yang lebih akurat dan mendalam, serta jawaban yang lebih relevan dan tepat.",
            "Penggunaan BERT dalam pengolahan data teks telah mengubah paradigma pemrosesan bahasa alami. Dengan memperhitungkan konteks sebelum dan sesudah sebuah kata, BERT memberikan representasi teks yang lebih kaya dan akurat. Dalam analisis sentimen, BERT membantu memahami nuansa emosional dan konteks sosial. Dalam pemahaman pertanyaan-jawaban, BERT meningkatkan respons terhadap pertanyaan kompleks, memperluas aplikasi dalam pencarian informasi dan asisten virtual."
        ]
    },
    
]



import streamlit as st

if "selected_prompt" not in st.session_state:
    st.session_state.selected_prompt = None
    
from PIL import Image
img = Image.open('D:\\Kuliah\BISMILLAH TAMAT TEPAT WAKTU SEBELUM BULAN 5\\WebsiteEssay\\LOGO AES.png')



# Define UI
st.set_page_config(
    page_title="EssayGrading",
    page_icon=img, # Optional, if you have a favicon
    layout="wide",
    initial_sidebar_state="collapsed" # Optional, change to "expanded" if you want sidebar expanded by default
)

# Custom CSS to set background color to beige
st.markdown("""
    <style>
        body {
            background-color: #f5f5dc;
        }
        .logo {
            display: flex;
            align-items: center;
        }
        .logo img {
            width: 50px;
            height: auto;
            margin-right: 10px;
        }
        .center {
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

#st.image(img, width=10, use_column_width=True)
st.markdown("<h1 style='text-align: center;'>ESSAYGRADER</h1>", unsafe_allow_html=True)

# Tambahkan emoji dan tautan ke halaman utama di sidebar
#st.sidebar.markdown("📝 [Main Page](#)")

# Sekarang Anda dapat menggunakan session state dengan aman
def main_page():
    if st.session_state.selected_prompt is None:
        st.write('<div class="center">Silakan pilih soal:</div>', unsafe_allow_html=True)
        for i, prompt in enumerate(prompts, start=1):
            # Memberikan kunci unik ke tombol dengan menggunakan indeks iterasi
            if st.button(f"Soal {i}: {prompt['prompt']}", key=f"button_{i}"):
                st.session_state.selected_prompt = prompt
    else:
        score_essay(st.session_state.selected_prompt)


def score_essay(prompt):
    # Tampilkan soal esai
    st.write(f"<div class='center'>Soal: {prompt['prompt']}</div>", unsafe_allow_html=True)

    # Tampilkan kolom teks untuk memasukkan esai
    essay = st.text_area("Masukkan esai Anda:", height=300)  # Tinggi kolom diperbesar menjadi 300 piksel

    # Tombol "Get Score"
    if st.button("Get Score"):
        if essay:
            prompt_text = prompt["prompt"]
            reference_essays = prompt["reference_essays"]

            # Hitung skor
            final_score, feedback_text = predict_essay_score(essay, prompt_text, reference_essays, model, tokenizer, cuda, word2vec_model, bigru_model)
            st.write(f"Skor Akhir: {final_score}")
            st.write(f"Umpan Balik: {feedback_text}")
        else:
            st.warning("Harap masukkan esai terlebih dahulu.")

    # Tombol "Kembali"
    if st.button("Kembali"):
        st.session_state.selected_prompt = None
        
# Jalankan aplikasi
#main_page()

col1, col2, col3 , col4, col5 = st.columns([1, 1, 2, 1, 1])

with col1:
    pass

with col2:
    pass

with col4:
    pass

with col5:
    pass

with col3:
    main_page()

