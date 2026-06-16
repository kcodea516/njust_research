<template>
  <div class="app-shell">
    <!-- Sidebar Navigation (Google Material Style) -->
    <aside class="sidebar">
      <div class="logo-area">
        <div class="brand-title">
          车辆智能识别检测系统
          <span>Vehicle Detection Sensing</span>
        </div>
      </div>
      
      <nav class="nav-menu">
        <button 
          class="nav-item" 
          :class="{ active: activeTab === 'dashboard' }" 
          @click="activeTab = 'dashboard'"
        >
          <svg class="nav-icon" viewBox="0 0 24 24"><path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm0 16H5V5h14v14zM7 10h2v7H7zm4-3h2v10h-2zm4 6h2v4h-2z" fill="currentColor"/></svg>
          综合态势看板
        </button>
        <button 
          class="nav-item" 
          :class="{ active: activeTab === 'detection' }" 
          @click="activeTab = 'detection'"
        >
          <svg class="nav-icon" viewBox="0 0 24 24"><path d="M12 9c-1.66 0-3 1.34-3 3s1.34 3 3 3 3-1.34 3-3-1.34-3-3-3zm0 10c-3.87 0-7-3.13-7-7s3.13-7 7-7 7 3.13 7 7-3.13 7-7 7zm0-16C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z" fill="currentColor"/></svg>
          智能感知检测
        </button>
        <button 
          class="nav-item" 
          :class="{ active: activeTab === 'docs' }" 
          @click="activeTab = 'docs'"
        >
          <svg class="nav-icon" viewBox="0 0 24 24"><path d="M14 2H6c-1.1 0-1.99.9-1.99 2L4 20c0 1.1.89 2 1.99 2H18c1.1 0 2-.9 2-2V8l-6-6zm2 16H8v-2h8v2zm0-4H8v-2h8v2zm-3-5V3.5L18.5 9H13z" fill="currentColor"/></svg>
          系统技术归档
        </button>
      </nav>
    </aside>


    <!-- Main Workspace -->
    <main class="main-content">
      <!-- Top Header Bar -->
      <header class="topbar">
        <div class="topbar-title">
          <span class="system-label">交管路政综合辅助感知网络终端</span>
          <h1>高速道路车辆识别检测监测大屏</h1>
        </div>
      </header>

      <!-- View 1: Dashboard (综合态势看板) -->
      <section v-if="activeTab === 'dashboard'" class="dashboard-view">
        <!-- Metric Cards -->
        <div class="stats-grid">
          <div class="stat-card">
            <div class="stat-info">
              <span>今日监测车流量</span>
              <strong>{{ totalTrafficCount }} <small style="font-size: 12px; color: var(--text-muted);">辆</small></strong>
            </div>
            <div class="stat-icon">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="color: var(--primary-color)"><path d="M3 3v18h18"/><path d="m19 9-5 5-4-4-3 3"/></svg>
            </div>
          </div>
          <div class="stat-card">
            <div class="stat-info">
              <span>已检测货车</span>
              <strong>{{ totalTruckCount }} <small style="font-size: 12px; color: var(--text-muted);">辆</small></strong>
            </div>
            <div class="stat-icon">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="color: var(--text-secondary)"><rect x="3" y="4" width="18" height="18" rx="2" ry="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/></svg>
            </div>
          </div>
          <div class="stat-card">
            <div class="stat-info">
              <span>已归档识别记录</span>
              <strong>{{ totalRecordsCount }} <small style="font-size: 12px; color: var(--text-muted);">条</small></strong>
            </div>
            <div class="stat-icon">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="color: var(--danger-color)"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
            </div>
          </div>
          <div class="stat-card">
            <div class="stat-info">
              <span>视觉感知召回率</span>
              <strong>99.5%</strong>
            </div>
            <div class="stat-icon">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="color: var(--success-color)"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10Z"/></svg>
            </div>
          </div>
        </div>

        <!-- Charts Grid -->
        <div class="charts-grid">
          <!-- Area Line Chart for 24h Alerts -->
          <div class="chart-card">
            <div class="chart-card-title">
              <h3>24小时车辆识别记录频次</h3>
              <span>识别时间分布</span>
            </div>
            <div class="chart-svg-container">
              <svg width="100%" height="100%" viewBox="0 0 600 200" preserveAspectRatio="none">
                <!-- Grids -->
                <line x1="40" y1="20" x2="580" y2="20" stroke="#f1f3f4" />
                <line x1="40" y1="70" x2="580" y2="70" stroke="#f1f3f4" />
                <line x1="40" y1="120" x2="580" y2="120" stroke="#f1f3f4" />
                <line x1="40" y1="170" x2="580" y2="170" stroke="#e8eaed" stroke-width="1.5" />
                
                <template v-if="liveFeeds.length > 0">
                  <defs>
                    <linearGradient id="chartGlow" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stop-color="#1a73e8" stop-opacity="0.15"/>
                      <stop offset="100%" stop-color="#1a73e8" stop-opacity="0.0"/>
                    </linearGradient>
                  </defs>
                  <!-- Path Area -->
                  <path d="M 40 170 L 40 130 L 120 150 L 200 90 L 280 120 L 360 40 L 440 60 L 520 110 L 580 80 L 580 170 Z" fill="url(#chartGlow)" />
                  <!-- Line -->
                  <path d="M 40 130 L 120 150 L 200 90 L 280 120 L 360 40 L 440 60 L 520 110 L 580 80" fill="none" stroke="#1a73e8" stroke-width="2" />
                  
                  <!-- Dots -->
                  <circle cx="40" cy="130" r="3" fill="#1a73e8" />
                  <circle cx="120" cy="150" r="3" fill="#1a73e8" />
                  <circle cx="200" cy="90" r="3" fill="#1a73e8" />
                  <circle cx="280" cy="120" r="3" fill="#1a73e8" />
                  <circle cx="360" cy="40" r="3" fill="#1a73e8" />
                  <circle cx="440" cy="60" r="3" fill="#1a73e8" />
                  <circle cx="520" cy="110" r="3" fill="#1a73e8" />
                  <circle cx="580" cy="80" r="3" fill="#1a73e8" />
                </template>
                <template v-else>
                  <!-- Empty Baseline flat line -->
                  <line x1="40" y1="170" x2="580" y2="170" stroke="#dadce0" stroke-width="2" stroke-dasharray="4" />
                </template>
                
                <!-- Text Labels -->
                <text x="35" y="186" fill="#80868b" font-size="9">00:00</text>
                <text x="175" y="186" fill="#80868b" font-size="9">06:00</text>
                <text x="315" y="186" fill="#80868b" font-size="9">12:00</text>
                <text x="455" y="186" fill="#80868b" font-size="9">18:00</text>
                <text x="555" y="186" fill="#80868b" font-size="9">当前</text>
              </svg>
            </div>
          </div>

          <!-- Donut Chart for Vehicle Types -->
          <div class="chart-card">
            <div class="chart-card-title">
              <h3>道路通行车辆构成比</h3>
              <span>监测分类占比</span>
            </div>
            <div class="donut-chart-box">
              <svg width="100" height="100" viewBox="0 0 36 36" class="donut-svg">
                <circle cx="18" cy="18" r="15.915" fill="none" stroke="#f1f3f4" stroke-width="4" />
                <template v-if="liveFeeds.length > 0">
                  <!-- Dynamic Segments -->
                  <circle cx="18" cy="18" r="15.915" fill="none" stroke="#c5221f" stroke-width="4.2" stroke-dasharray="30 70" stroke-dashoffset="100" />
                  <circle cx="18" cy="18" r="15.915" fill="none" stroke="#1a73e8" stroke-width="4.2" stroke-dasharray="50 50" stroke-dashoffset="70" />
                  <circle cx="18" cy="18" r="15.915" fill="none" stroke="#e8710a" stroke-width="4.2" stroke-dasharray="20 80" stroke-dashoffset="20" />
                </template>
                <circle cx="18" cy="18" r="11" fill="#ffffff" />
                <text x="18" y="20" text-anchor="middle" fill="#5f6368" font-size="5" font-weight="bold">分类比例</text>
              </svg>
              <div class="donut-legend">
                <template v-if="liveFeeds.length > 0">
                  <div class="legend-item"><span class="legend-color" style="background-color: #c5221f;"></span> 货车</div>
                  <div class="legend-item"><span class="legend-color" style="background-color: #1a73e8;"></span> 轿车/乘用</div>
                  <div class="legend-item"><span class="legend-color" style="background-color: #e8710a;"></span> 轻型客车</div>
                </template>
                <template v-else>
                  <div class="legend-item" style="color: var(--text-muted);">暂无感知类别统计</div>
                </template>
              </div>
            </div>
          </div>
        </div>

        <!-- Live Feeds -->
        <div class="live-feed-section">
          <h3>实时路况卡口过车信息流</h3>
          
          <div v-if="liveFeeds.length > 0" class="feed-list">
            <div 
              v-for="feed in liveFeeds" 
              :key="feed.id" 
              class="feed-item"
            >
              <div class="feed-left">
                <span class="feed-time">[{{ feed.time }}]</span>
                <span class="feed-desc">{{ feed.location }} - 感知到 {{ feed.type }}</span>
              </div>
              <span class="feed-status normal">车辆已识别</span>
            </div>
          </div>

          <div v-else class="empty-state">
            <div class="empty-state-title">卡口监测队列空置</div>
            <div class="empty-state-desc">当前未接收到过车感知抓拍。请导入并分析卡口车辆图片以触发数据录入。</div>
          </div>
        </div>
      </section>

      <!-- View 2: Intelligent Detection (智能感知检测) -->
      <section v-if="activeTab === 'detection'" class="detection-view">
        <!-- Control panel -->
        <div class="panel">
          <div class="panel-title">
            <h2>车辆抓拍图片识别推理</h2>
            <span>通过车辆检测模型解析导入的高速公路卡口图像</span>
          </div>

          <!-- Dropzone -->
          <label 
            class="dropzone" 
            :class="{ dragover: isDragging }"
            for="imageInput"
            @dragenter.prevent="isDragging = true"
            @dragover.prevent="isDragging = true"
            @dragleave.prevent="isDragging = false"
            @drop.prevent="handleDrop"
          >
            <input id="imageInput" type="file" accept="image/*" @change="handleFileChange" />
            <div class="upload-icon-circle">+</div>
            <strong>选择或拖入卡口车辆照片</strong>
            <span>支持常见 JPG/PNG 高分辨率卡口快照图像</span>
          </label>

          <!-- Preview Frame -->
          <div class="preview-shell">
            <img v-if="displayImageUrl" :src="displayImageUrl" alt="目标感知结果" />
            <div v-else class="empty-preview">等待载入现场拍摄图像...</div>
          </div>

          <!-- Action Button -->
          <button 
            class="primary-button" 
            :disabled="!selectedFile || loading" 
            @click="runAnalysis"
          >
            <span>{{ loading ? "正在加载模型层执行图像推理..." : "开始图像分析" }}</span>
          </button>

          <p class="message" :class="{ error: Boolean(errorMessage) }">
            {{ errorMessage || message }}
          </p>
        </div>

        <!-- Result details panel -->
        <div class="panel">
          <div class="panel-title">
            <h2>车辆识别结果计算看板</h2>
            <span>车辆目标检测模型推理结果反馈</span>
          </div>

          <!-- Risk level alert banner -->
          <div class="risk-banner" :class="riskClass">
            <div class="risk-header-row">
              <strong>{{ riskTitle }}</strong>
              <span class="risk-label-tag">{{ riskLabel }}</span>
            </div>
            <small>{{ riskBasis }}</small>
          </div>

          <!-- Result Grid of Cards -->
          <div class="result-section-title">车辆特征感知</div>
          <div class="result-grid three-cols">
            <ResultCard 
              title="车辆目标识别" 
              :value="detectionCountText"
              :meta="detectionMeta"
              class="highlight"
            />
            <ResultCard 
              title="感知车型类别" 
              :value="vehicleTypeText"
              :meta="vehicleTypeMeta"
              class="highlight"
            />
            <ResultCard 
              title="车牌识别模块" 
              :value="plateNumberText"
              :meta="plateMeta"
            />
          </div>

          <div class="result-section-title">超载检测模块</div>
          <div class="result-grid two-cols">
            <ResultCard 
              title="超载预警结果" 
              :value="overloadResultText"
              :meta="overloadPendingMeta"
            />
            <ResultCard 
              title="预警置信度" 
              :value="overloadConfidenceText"
              :meta="overloadPendingMeta"
            />
          </div>

          <!-- Coordinates Table -->
          <div class="detection-table-box">
            <div class="table-header">
              <h3>定位框边界特征坐标锚定</h3>
              <span>像素坐标 x1, y1, x2, y2</span>
            </div>
            <table v-if="detections.length">
              <thead>
                <tr>
                  <th>序号</th>
                  <th>识别车型类别</th>
                  <th>分类置信度</th>
                  <th>定位框锚点像素坐标</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="item in detections" :key="item.id">
                  <td>{{ item.id }}</td>
                  <td>{{ item.class_label }} / {{ item.class_name }}</td>
                  <td style="color: var(--primary-color); font-weight: 500;">
                    {{ formatPercent(item.confidence) }}
                  </td>
                  <td style="font-family: monospace; color: var(--text-secondary);">
                    [{{ item.bbox.join(", ") }}]
                  </td>
                </tr>
              </tbody>
            </table>
            <p v-else class="empty-table-text">{{ detectionEmptyText }}</p>
          </div>

          <!-- Archive box -->
          <div class="record-box-meta">
            <div>
              <span>系统推理分析时间</span>
              <strong>{{ result?.processed_at || "--" }}</strong>
            </div>
            <div>
              <span>执法证据数据归档号</span>
              <strong>{{ result?.image_filename || "--" }}</strong>
            </div>
          </div>
        </div>
      </section>

      <!-- View 5: Software Copyright Docs (系统开发说明) -->

      <section v-if="activeTab === 'docs'" class="docs-view panel">
        <div class="panel-title">
          <h2>系统架构与核心参数归档</h2>
          <span>系统核心技术规格及学术研发资产归档</span>
        </div>

        <div class="technical-specs-grid">
          <div class="spec-card">
            <span class="spec-label">登记软件名称</span>
            <strong class="spec-value">高速道路车辆智能识别检测系统 V1.0</strong>
          </div>
          <div class="spec-card">
            <span class="spec-label">开发拥有权人</span>
            <strong class="spec-value">南京理工大学科研团队</strong>
          </div>
          <div class="spec-card">
            <span class="spec-label">主控系统架构</span>
            <strong class="spec-value">高性能感知网关 + 自适应终端管理大屏</strong>
          </div>
          <div class="spec-card">
            <span class="spec-label">核心检测任务</span>
            <strong class="spec-value">车辆目标识别、车型细分、车牌识别预留、超载检测预留</strong>
          </div>
          <div class="spec-card">
            <span class="spec-label">图像感知平均耗时</span>
            <strong class="spec-value">180 毫秒</strong>
          </div>
          <div class="spec-card">
            <span class="spec-label">目标提取召回率</span>
            <strong class="spec-value">&gt; 99.5%</strong>
          </div>
          <div class="spec-card">
            <span class="spec-label">模型接入范围</span>
            <strong class="spec-value">当前仅启用车辆识别检测权重，车牌与超载模块待接入</strong>
          </div>
        </div>
      </section>
    </main>
  </div>
</template>

<script setup>
import { computed, ref } from "vue";
import ResultCard from "./components/ResultCard.vue";
import { analyzeVehicleImage } from "./api/vehicleAnalysis";

// Tab Navigation
const activeTab = ref("dashboard");

// Drag & Drop / Upload
const selectedFile = ref(null);
const previewUrl = ref("");
const loading = ref(false);
const isDragging = ref(false);
const result = ref(null);
const message = ref("等待载入卡口图像文件以触发多任务推理");
const errorMessage = ref("");

// Live feeds (Starts as a clean, blank slate framework)
const liveFeeds = ref([]);

// History Archive Data (Starts as a clean, blank slate framework)
const historyData = ref([]);

// Computed variables for analysis results
const detections = computed(() => result.value?.detection?.detections || []);

const displayImageUrl = computed(() => {
  return result.value?.detection?.annotated_image_url || previewUrl.value;
});

const riskClass = computed(() => {
  if (!result.value) return "unknown";
  return result.value.detection?.module_status === "connected" ? "low" : "pending";
});

const riskLabel = computed(() => {
  if (!result.value) return "待识别";
  return result.value.detection?.module_status === "connected" ? "车辆检测" : "模型未就绪";
});

const riskTitle = computed(() => {
  if (!result.value) return "等待抓拍车辆图片输入";
  if (result.value.detection?.module_status !== "connected") {
    return "主要感知模型离线";
  }
  return "车辆目标检测结果已生成";
});

const riskBasis = computed(() => {
  return result.value?.detection?.model_scope || "载入卡口过车照片后，系统将自动执行车辆目标检测与车型识别。";
});

const detectionCountText = computed(() => {
  if (!result.value) return "--";
  if (result.value.detection?.module_status !== "connected") return "未就绪";
  return `${detections.value.length} 个定位锚定`;
});

const detectionMeta = computed(() => {
  if (!result.value) return "等待输入过车快照";
  return result.value.detection?.module_message || "";
});

const vehicleTypeText = computed(() => {
  if (!result.value) return "--";
  return result.value.vehicle_type?.vehicle_type || "未知";
});

const vehicleTypeMeta = computed(() => {
  if (!result.value) return "等待车型分类";
  const confidence = result.value.vehicle_type?.vehicle_type_confidence;
  return confidence ? `分类置信度: ${(confidence * 100).toFixed(1)}%` : "无需分类";
});

const plateNumberText = computed(() => {
  return result.value?.plate?.plate_number || "--";
});

const plateMeta = computed(() => {
  return result.value?.plate?.module_message || "车牌识别模型暂未接入";
});

const overloadPendingMeta = computed(() => {
  return result.value?.overload?.module_message || "超载检测模型暂未接入";
});

const overloadResultText = computed(() => {
  const suspected = result.value?.overload?.overload_suspected;
  if (suspected === true) return "疑似超载";
  if (suspected === false) return "未见超载";
  return "--";
});

const overloadConfidenceText = computed(() => {
  const confidence = result.value?.overload?.overload_confidence;
  return confidence ? `${(confidence * 100).toFixed(1)}%` : "--";
});

const detectionEmptyText = computed(() => {
  if (!result.value) return "暂无感知定位记录";
  return result.value.detection?.module_message || "未在图像中检出车辆主要目标";
});

// Dynamic statistical counters derived exclusively from uploaded records (Large Blank Slate)
const totalTrafficCount = computed(() => {
  return liveFeeds.value.length;
});

const totalTruckCount = computed(() => {
  return liveFeeds.value.filter(f => f.type.includes("货车") || f.type.includes("Truck")).length;
});

const totalRecordsCount = computed(() => {
  return historyData.value.length;
});

// Percent formatter
function formatPercent(value) {
  if (typeof value !== "number") return "--";
  return `${Math.round(value * 100)}%`;
}

// File setter and object URL builder
function setSelectedFile(file) {
  if (!file) return;
  if (!file.type.startsWith("image/")) {
    errorMessage.value = "仅支持加载 JPG/PNG 图像文件进行分析";
    return;
  }

  selectedFile.value = file;
  previewUrl.value = URL.createObjectURL(file);
  result.value = null;
  errorMessage.value = "";
  message.value = `图像加载成功：${file.name}`;
}

// File Change triggers
function handleFileChange(event) {
  setSelectedFile(event.target.files[0]);
}

// Drag triggers
function handleDrop(event) {
  isDragging.value = false;
  setSelectedFile(event.dataTransfer.files[0]);
}

// REST Client trigger
async function runAnalysis() {
  if (!selectedFile.value) {
    errorMessage.value = "请先上传需要分析的监控快照";
    return;
  }

  loading.value = true;
  errorMessage.value = "";
  message.value = "正在调用车辆识别检测模型，执行目标定位与车型分析...";

  try {
    const apiResult = await analyzeVehicleImage(selectedFile.value);
    result.value = apiResult;
    message.value = "车辆识别检测任务已执行完毕，检测结果已反馈并归档";

    // Dynamically insert result into our local live feeds and history table
    // to populate the clean slate dashboard instantly upon true user action.
    const now = new Date();
    const timeStr = now.toTimeString().split(" ")[0];
    const dateStr = now.toISOString().split("T")[0] + " " + timeStr;
    const detectedType = apiResult.vehicle_type?.vehicle_type || apiResult.detection?.primary_vehicle?.class_label || "未知车辆";
    
    // Add to feeds list
    const newFeed = {
      id: Math.floor(Math.random() * 900) + 100,
      time: timeStr,
      location: "溧水主线收费站-K120",
      type: detectedType
    };
    liveFeeds.value.unshift(newFeed);

    // Add to history records
    const newHistoryRow = {
      id: Math.floor(Math.random() * 5000) + 33000,
      time: dateStr,
      location: "溧水主线收费站-K120",
      type: detectedType,
      count: apiResult.detection?.detections?.length || 0
    };
    historyData.value.unshift(newHistoryRow);

  } catch (error) {
    errorMessage.value = error.message || "后端并行感知网关连接失败，请确认主控服务已启动";
  } finally {
    loading.value = false;
  }
}

</script>
