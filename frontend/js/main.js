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
    errorMessage.classList.remove("active");
    resultOutput.style.color = 'var(--text-color)'; /* Reset color */

    if (!sourceNumber) {
        errorMessage.textContent = '元の数字を入力してください。';
        errorMessage.classList.add("active");
        resultOutput.textContent = '';
        return;
    }
    if (!targetValue || isNaN(parseInt(targetValue))) {
        errorMessage.textContent = '目標値（自然数）を入力してください。';
        errorMessage.classList.add("active");
        resultOutput.textContent = '';
        return;
    }
    if (parseInt(targetValue) <= 0) {
        errorMessage.textContent = '目標値は正の自然数である必要があります。';
        errorMessage.classList.add("active");
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
                    if (window.innerWidth <= 600) {
                        resultText += `${step.remaining_before} を表現するために ${step.value_used} = ${step.expression} を使用。\n残り: ${step.remaining_after}\n`;
                    } else {
                        resultText += `${step.remaining_before} を表現するために ${step.value_used} = ${step.expression} を使用。残り: ${step.remaining_after}\n`;
                    }
                } else if (step.remaining_before > 0) {
                    resultText += `  これ以上分解できなかったよ。最終的な残り: ${step.remaining_before}\n`;
                }
            });
            if (data.full_expression_string) {
                resultText += `\n結果: ${data.total_expressed_value} = ${data.full_expression_string}`;
                if (data.final_remaining > 0) {
                    if (window.innerWidth <= 600) {
                        resultText += `\n+ (残り ${data.final_remaining})`;
                    } else {
                        resultText += ` + (残り ${data.final_remaining})`;
                    }
                }
            } else if (data.final_remaining > 0) {
                resultText += `\nおおよその表現: (残り ${data.final_remaining})`;
            } else {
                resultText += `\n完全に表現できた！`;
            }

            resultOutput.textContent = resultText;
            resultOutput.style.color = 'var(--primary-color)'; /* Highlight success */

        } else {
            errorMessage.textContent = `エラー: ${data.detail || '不明なエラー'}`;
            errorMessage.classList.add("active");
            resultOutput.textContent = '';
        }
    } catch (error) {
        errorMessage.textContent = `ネットワークエラー: ${error.message}`;
        errorMessage.classList.add("active");
        resultOutput.textContent = '';
    }
}

async function evaluateUserEquation() {
    const sourceNumber = sourceNumberInput.value.trim();
    const targetValue = targetValueInput.value.trim();
    const userEquation = targetEquationInput.value.trim();

    resultOutput2.textContent = '';
    errorMessage.textContent = '';
    errorMessage.classList.remove("active");

    if (!sourceNumber) {
        errorMessage.textContent = '元の数字を入力してください。';
        errorMessage.classList.add("active");
        return;
    }
    if (!targetValue || isNaN(parseInt(targetValue))) {
        errorMessage.textContent = '目標値（自然数）を入力してください。';
        errorMessage.classList.add("active");
        return;
    }
    if (!userEquation) {
        errorMessage.textContent = '式を入力してください。';
        errorMessage.classList.add("active");
        return;
    }

    try {
        const response = await fetch('/evaluate_user_expression', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                source_number_str: sourceNumber,
                target_value: parseInt(targetValue),
                user_expression: userEquation
            }),
        });
        const data = await response.json();

        if (response.ok) {
            let userResult = `【あなたの式の評価】\n`;
            userResult += `式: ${data.user_expression}\n`;
            userResult += `値: ${data.user_expression_value ?? '評価不可'}\n`;
            userResult += `長さ: ${data.user_expression_length}文字, 演算子数: ${data.user_operator_count}\n`;
            userResult += `判定: ${data.is_valid_expression ? (data.is_correct_value ? '正解！' : '値が一致しません') : '式が無効です'}\n`;
            if (data.comparison_result) {
                userResult += `比較: ${data.comparison_result}\n`;
            }
            resultOutput2.textContent = userResult;
        } else {
            errorMessage.textContent = `エラー: ${data.detail || '不明なエラー'}`;
            errorMessage.classList.add("active");
        }
    } catch (error) {
        errorMessage.textContent = `ネットワークエラー: ${error.message}`;
        errorMessage.classList.add("active");
    }
}

window.addEventListener('DOMContentLoaded', function() {
    const audio = document.getElementById('bgm');
    // ユーザー操作なしで自動再生はブラウザで制限される場合が多いので、下記の工夫をします
    function tryPlay() {
        audio.play().catch(() => {});
    }
    // ページをクリックしたら再生
    document.body.addEventListener('click', tryPlay, { once: true });
    // すぐに再生を試みる（許可されていれば自動再生される）
    tryPlay();
});