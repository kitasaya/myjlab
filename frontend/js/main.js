const sourceNumberInput = document.getElementById('sourceNumberInput');
const targetValueInput = document.getElementById('targetValueInput');
const resultOutput = document.getElementById('resultOutput');
const errorMessage = document.getElementById('errorMessage');
const targetEquationInput = document.getElementById('targetEquationInput');

async function processNumbers() {
    const sourceNumber = sourceNumberInput.value.trim();
    const targetValue = targetValueInput.value.trim();

    resultOutput.textContent = '処理中...';
    errorMessage.textContent = '';
    resultOutput.style.color = 'var(--text-color)'; /* Reset color */

    if (!sourceNumber) {
        errorMessage.textContent = '元の数字を入力してください。';
        resultOutput.textContent = '';
        return;
    }
    if (!targetValue || isNaN(parseInt(targetValue))) {
        errorMessage.textContent = '目標値（自然数）を入力してください。';
        resultOutput.textContent = '';
        return;
    }
    if (parseInt(targetValue) <= 0) {
        errorMessage.textContent = '目標値は正の自然数である必要があります。';
        resultOutput.textContent = '';
        return;
    }

    try {
        const response = await fetch('/process_number', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                source_number_str: sourceNumber,
                target_value: parseInt(targetValue)
            }),
        });

        const data = await response.json();

        if (response.ok) {
            let resultText = `目標値 ${targetValue} の分解結果:\n\n`;
            data.decomposition_steps.forEach(step => {
                if (step.expression !== 'No further decomposition') {
                    resultText += `  ${step.remaining_before} を表現するために ${step.value_used} = ${step.expression} を使用。残り: ${step.remaining_after}\n`;
                } else if (step.remaining_before > 0) {
                    resultText += `  これ以上分解できませんでした。最終的な残り: ${step.remaining_before}\n`;
                }
            });
            if (data.full_expression_string) {
                resultText += `\n結果: ${data.total_expressed_value} = ${data.full_expression_string}`;
                if (data.final_remaining > 0) {
                    resultText += ` + (残り ${data.final_remaining})`;
                }
            } else if (data.final_remaining > 0) {
                resultText += `\nおおよその表現: (残り ${data.final_remaining})`;
            } else {
                resultText += `\n完全に表現できました！`;
            }

            resultOutput.textContent = resultText;
            resultOutput.style.color = 'var(--primary-color)'; /* Highlight success */

        } else {
            errorMessage.textContent = `エラー: ${data.detail || '不明なエラー'}`;
            resultOutput.textContent = '';
        }
    } catch (error) {
        errorMessage.textContent = `ネットワークエラー: ${error.message}`;
        resultOutput.textContent = '';
    }
}
