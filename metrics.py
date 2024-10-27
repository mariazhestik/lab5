import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Функція для відображення теплової карти
def display_heatmap(data, title):
    plt.figure(figsize=(6, 6))
    # Встановлюємо annot=False, щоб не відображати значення
    sns.heatmap(data, annot=False, cmap="coolwarm", square=True, cbar=True)
    plt.title(title)
    plt.show()

# Функція для побудови гістограми
def display_histogram(values, segment_sizes, title):
    plt.figure(figsize=(8, 6))
    plt.bar(segment_sizes, values, color="skyblue")
    plt.xlabel("Розмір сегмента")
    plt.ylabel("Середнє значення")
    plt.title(title)
    plt.show()

# Функція для завантаження зображення
def load_image(image_path):
    # Завантаження зображення в градаціях сірого
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Помилка: Не вдалося завантажити зображення за шляхом {image_path}")
    return image

# Функція для розбиття зображення на сегменти та їх візуалізації
def split_image_and_display_segments(image, segment_sizes):
    for size in segment_sizes:
        h, w = image.shape
        segmented_image = np.zeros_like(image)
        
        # Розбиття на сегменти та середнє значення пікселів для кожного сегмента
        for i in range(0, h, size):
            for j in range(0, w, size):
                segment = image[i:i+size, j:j+size]
                mean_value = int(np.mean(segment))
                segmented_image[i:i+size, j:j+size] = mean_value
        
        # Відображення сегментованого зображення
        plt.figure(figsize=(6, 6))
        plt.imshow(segmented_image, cmap="gray")
        plt.title(f"Сегментоване зображення ({size}x{size})")
        plt.axis("off")
        plt.show()

# Функція для аналізу серій однакових елементів у сегментах
def analyze_series_in_segments(image, segment_size):
    h, w = image.shape
    series_lengths = []
    
    for i in range(0, h, segment_size):
        for j in range(0, w, segment_size):
            segment = image[i:i+segment_size, j:j+segment_size]
            unique_values, counts = np.unique(segment, return_counts=True)
            series_lengths.append(counts.mean())  # Середня довжина серії
    return series_lengths

# Функція для підрахунку перепадів яскравості у сегментах
def count_brightness_transitions(image, segment_size):
    h, w = image.shape
    transitions_counts = []
    
    for i in range(0, h, segment_size):
        for j in range(0, w, segment_size):
            segment = image[i:i+segment_size, j:j+segment_size]
            transitions = np.sum(np.abs(np.diff(segment, axis=0))) + np.sum(np.abs(np.diff(segment, axis=1)))
            transitions_counts.append(transitions)  # Кількість перепадів у сегменті
    return transitions_counts

# Основна функція для виконання аналізу та візуалізації результатів
def main(image_path):
    # Крок 1: Завантаження зображення
    image = load_image(image_path)
    if image is None:
        return  # Зупинити виконання, якщо зображення не було завантажено
    
    # Крок 2: Розбиття на сегменти та візуалізація
    segment_sizes = [8, 16, 32, 64, 128]
    split_image_and_display_segments(image, segment_sizes)
    
    # Створимо списки для середніх значень, щоб побудувати гістограми
    avg_series_lengths = []
    avg_transitions_counts = []
    
    # Кроки 3 і 4: Аналіз серій однакових елементів та перепадів яскравості з візуалізацією
    for size in segment_sizes:
        series_lengths = analyze_series_in_segments(image, size)  # Завдання 1
        transitions_counts = count_brightness_transitions(image, size)  # Завдання 2
        
        # Розрахунок середніх значень
        avg_series = np.mean(series_lengths)
        avg_transitions = np.mean(transitions_counts)
        avg_series_lengths.append(avg_series)
        avg_transitions_counts.append(avg_transitions)
        
        # Перетворюємо результати на 2D масив для теплової карти
        h, w = image.shape
        heatmap_series = np.array(series_lengths).reshape(h // size, w // size)
        heatmap_transitions = np.array(transitions_counts).reshape(h // size, w // size)
        
        # Відображаємо теплові карти для серій і перепадів
        display_heatmap(heatmap_series, f'Довжина серій {size}x{size}')
        display_heatmap(heatmap_transitions, f'Перепади яскравості {size}x{size}')
    
    # Побудова гістограм для середніх значень за кожним розміром сегмента
    display_histogram(avg_series_lengths, segment_sizes, "Середня довжина серії (Завдання 1)")
    display_histogram(avg_transitions_counts, segment_sizes, "Середня кількість перепадів яскравості (Завдання 2)")

# Запуск з прикладом шляху до зображення
main('image/I22.BMP')  # або 'image\\I22.BMP'
