import { BrowserRouter, Route, Routes } from "react-router-dom";
import { ThemeProvider } from "@/contexts/theme-context";
import { Layout } from "@/components/layout/layout";
import { Home } from "@/pages/home";
import { ClinicalData } from "@/pages/clinical-data";
import { UploadImage } from "@/pages/upload-image";
import { Mammography } from "@/pages/mammography";
import { ModelPerformance } from "@/pages/model-performance";
import { ClusterExplorer } from "@/pages/cluster-explorer";
import { About } from "@/pages/about";

export default function App() {
  return (
    <ThemeProvider>
      <BrowserRouter>
        <Routes>
          <Route element={<Layout />}>
            <Route path="/" element={<Home />} />
            <Route path="/clinical-data" element={<ClinicalData />} />
            <Route path="/upload-image" element={<UploadImage />} />
            <Route path="/mammography" element={<Mammography />} />
            <Route path="/model-performance" element={<ModelPerformance />} />
            <Route path="/cluster-explorer" element={<ClusterExplorer />} />
            <Route path="/about" element={<About />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </ThemeProvider>
  );
}
