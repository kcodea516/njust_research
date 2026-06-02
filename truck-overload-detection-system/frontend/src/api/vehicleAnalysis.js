const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "";

export async function analyzeVehicleImage(file) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE_URL}/api/analyze`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || "识别服务暂不可用");
  }

  return response.json();
}
