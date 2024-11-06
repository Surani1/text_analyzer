document.getElementById('analyzeButton').addEventListener('click', analyzeText);

async function analyzeText() {
    const resultsDiv = document.getElementById('results');
    const plotDiv = document.getElementById('plot');
    resultsDiv.innerHTML = 'Анализ...';
    plotDiv.innerHTML = '';

    const formData = new FormData();
    const file = document.getElementById('file').files[0];
    const text = document.getElementById('text').value;

    formData.append(file ? 'file' : 'text', file || text);

    try {
        const response = await fetch('/analyze', {
            method: 'POST',
            body: formData
        });
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
        switch(key) {
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