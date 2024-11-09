const textArea = document.getElementById('text');
const fileInput = document.getElementById('file');
const removeFileButton = document.getElementById('removeFileButton');

document.querySelectorAll('input[name="inputType"]').forEach(radio => {
    radio.addEventListener('change', toggleInputMethod);
});
fileInput.addEventListener('change', handleFileChange);
removeFileButton.addEventListener('click', removeFile);
document.getElementById('analyzeButton').addEventListener('click', analyzeText);

function toggleInputMethod() {
    const selectedType = document.querySelector('input[name="inputType"]:checked').value;

    if (selectedType === 'text') {
        textArea.style.display = 'block';
        fileInput.style.display = 'none';
        removeFileButton.style.display = 'none';
        textArea.value = ''; // Очистить текстовое поле
    } else {
        textArea.style.display = 'none';
        fileInput.style.display = 'block';
        removeFileButton.style.display = 'block';
        fileInput.value = ''; // Очистить поле загрузки файла
    }
}

function handleFileChange() {
    removeFileButton.style.display = fileInput.files.length ? 'block' : 'none';
}

function removeFile() {
    fileInput.value = '';
    removeFileButton.style.display = 'none';
}

async function analyzeText() {
    const resultsDiv = document.getElementById('results');
    const plotDiv = document.getElementById('plot');
    resultsDiv.innerHTML = 'Анализ...';
    plotDiv.innerHTML = '';

    const formData = new FormData();
    const selectedType = document.querySelector('input[name="inputType"]:checked').value;

    if (selectedType === 'text') {
        const text = textArea.value;
        if (!text.trim()) {
            resultsDiv.innerHTML = '<p class="error">Пожалуйста, введите текст для анализа.</p>';
            return;
        }
        formData.append('text', text);
    } else {
        const file = fileInput.files[0];
        if (!file) {
            resultsDiv.innerHTML = '<p class="error">Пожалуйста, загрузите файл для анализа.</p>';
            return;
        }
        formData.append('file', file);
    }

    try {
        const response = await fetch('/analyze', { method: 'POST', body: formData });
        const data = await response.json();

        if (data.error) {
            resultsDiv.innerHTML = `<p class="error">${data.error}</p>`;
            return;
        }

        displayResults(data);
    } catch (error) {
        resultsDiv.innerHTML = `<p class="error">Ошибка: ${error.message}</p>`;
    }
}

function displayResults(data) {
    const resultsDiv = document.getElementById('results');
    const plotDiv = document.getElementById('plot');
    
    let html = '<h2>Результаты анализа:</h2>';

    for (const [key, value] of Object.entries(data)) {
        switch (key) {
            case 'plot':
                plotDiv.innerHTML = `<img src="data:image/png;base64,${value}" alt="Word frequency plot">`;
                break;
            case 'Ключевые слова':
                html += '<h3>Ключевые слова:</h3><ul>' + value.map(word => `<li>${word}</li>`).join('') + '</ul>';
                break;
            case 'Исправленный текст':
                document.getElementById('correctedText').innerHTML = `<h3>Исправленный текст:</h3><p>${value}</p>`;
                break;
            default:
                html += `<p><strong>${key}:</strong> ${value}</p>`;
        }
    }
    
    resultsDiv.innerHTML = html; 
}

document.getElementById('showCorrectedText').addEventListener('change', () => {
    const correctedTextDiv = document.getElementById('correctedText');
    correctedTextDiv.classList.toggle('hidden', !document.getElementById('showCorrectedText').checked);
});

// Initial setup
toggleInputMethod();