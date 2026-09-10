const dropZone = document.getElementById("drop-zone");
const imageInput = document.getElementById("image-input");
const originalImage = document.getElementById("original-image");

let selectedFile = null;
let previewUrl = null;


// Drop Zone 클릭
dropZone.addEventListener("click", () => {
    imageInput.click();
});


// 파일 선택
imageInput.addEventListener("change", () => {
    const file = imageInput.files[0];

    if (!file) {
        return;
    }

    handleFile(file);
});


// 파일을 Drop Zone 위로 가져왔을 때
dropZone.addEventListener("dragover", (event) => {
    event.preventDefault();

    dropZone.classList.add("drag-over");
});


// 파일이 Drop Zone 밖으로 나갔을 때
dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("drag-over");
});


// 파일을 Drop 했을 때
dropZone.addEventListener("drop", (event) => {
    event.preventDefault();

    dropZone.classList.remove("drag-over");

    const file = event.dataTransfer.files[0];

    if (!file) {
        return;
    }

    handleFile(file);
});


// 선택된 파일 처리
function handleFile(file) {

    if (!file.type.startsWith("image/")) {
        alert("이미지 파일만 업로드할 수 있습니다.");
        return;
    }

    selectedFile = file;

    // 기존 Preview URL 제거
    if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
    }

    // 새로운 Preview URL 생성
    previewUrl = URL.createObjectURL(file);

    // Original 이미지 표시
    originalImage.src = previewUrl;
}


const segmentButton = document.getElementById("segment-button");

segmentButton.addEventListener("click", async () => {
    if (!selectedFile) {
        alert("이미지를 선택해주세요.");
        return;
    }

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
        const response = await fetch("/api/segment", {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            throw new Error("이미지 처리에 실패했습니다.");
        }

        const result = await response.json();
        // console.log(result);
        resultImage.src = result.overlay_url;

    } catch (error) {
        console.error(error);
        alert("서버 요청 중 오류가 발생했습니다.");
    }
});