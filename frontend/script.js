const api = "http://127.0.0.1:8000/recommend";

document.getElementById("btn").onclick = async () => {
  const q = document.getElementById("query").value.trim();
  const k = parseInt(document.getElementById("topk").value) || 7;

  if (!q) return alert("Enter a query");

  document.getElementById("status").innerText = "Searching...";
  const tbody = document.querySelector("#results tbody");
  tbody.innerHTML = "";

  const res = await fetch(api, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({query: q, top_k: k})
  });

  const data = await res.json();
  document.getElementById("status").innerText = `Results for: ${data.query}`;

  data.recommendations.forEach((r, i) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${i+1}</td>
      <td><a href="${r.canonical_url}" target="_blank">${r.assessment_name}</a></td>
      <td>${r.test_type}</td>
      <td>${r.skills_tags}</td>
      <td>${r.score.toFixed(3)}</td>`;
    tbody.appendChild(tr);
  });
};
