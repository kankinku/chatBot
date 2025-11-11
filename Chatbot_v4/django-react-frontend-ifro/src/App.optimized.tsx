import React, { Suspense } from "react";
import { Routes, Route, Navigate, Outlet } from "react-router-dom";
import { MainLayout } from "./shared/components/layout";
import { ProgressIndicator } from "./shared/components/ui";

// Lazy load components for better performance
const Dashboard = React.lazy(
  () => import("./features/dashboard/components/index")
);
const AdminDashboard = React.lazy(() =>
  import("./features/dashboard").then((module) => ({
    default: module.AdminDashboard,
  }))
);
const LoginForm = React.lazy(
  () => import("./features/auth/components/LoginForm")
);
const RegisterForm = React.lazy(
  () => import("./features/auth/components/RegisterForm")
);
const Settings = React.lazy(() => import("./shared/components/Settings"));

// Policy Proposals - Lazy loaded
const ProposalList = React.lazy(() =>
  import("./features/policy-proposals").then((module) => ({
    default: module.ProposalList,
  }))
);
const CreateProposalForm = React.lazy(() =>
  import("./features/policy-proposals").then((module) => ({
    default: module.CreateProposalForm,
  }))
);
const EditProposalForm = React.lazy(() =>
  import("./features/policy-proposals").then((module) => ({
    default: module.EditProposalForm,
  }))
);
const ProposalDetail = React.lazy(() =>
  import("./features/policy-proposals").then((module) => ({
    default: module.ProposalDetail,
  }))
);
const AdminProposalManagement = React.lazy(() =>
  import("./features/policy-proposals").then((module) => ({
    default: module.AdminProposalManagement,
  }))
);

const PrivateRoute = () => {
  const isAuth = !!localStorage.getItem("access");
  return isAuth ? <Outlet /> : <Navigate to="/login" />;
};

const LayoutWrapper = ({ children }: { children: React.ReactNode }) => {
  return <MainLayout>{children}</MainLayout>;
};

const LoadingFallback = () => (
  <div className="flex items-center justify-center min-h-screen">
    <ProgressIndicator progress={50} />
  </div>
);

function App() {
  return (
    <Suspense fallback={<LoadingFallback />}>
      <Routes>
        <Route path="/login" element={<LoginForm />} />
        <Route path="/register" element={<RegisterForm />} />
        <Route element={<PrivateRoute />}>
          {/* 대시보드는 기존 디자인 유지 */}
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/admin" element={<AdminDashboard />} />

          {/* 정책제안과 설정은 새로운 레이아웃 사용 */}
          <Route
            path="/proposals"
            element={
              <LayoutWrapper>
                <ProposalList />
              </LayoutWrapper>
            }
          />
          <Route
            path="/proposals/create"
            element={
              <LayoutWrapper>
                <CreateProposalForm />
              </LayoutWrapper>
            }
          />
          <Route
            path="/proposals/:id/edit"
            element={
              <LayoutWrapper>
                <EditProposalForm />
              </LayoutWrapper>
            }
          />
          <Route
            path="/proposals/:id"
            element={
              <LayoutWrapper>
                <ProposalDetail />
              </LayoutWrapper>
            }
          />
          <Route
            path="/admin/proposals"
            element={
              <LayoutWrapper>
                <AdminProposalManagement />
              </LayoutWrapper>
            }
          />
          <Route
            path="/settings"
            element={
              <LayoutWrapper>
                <Settings />
              </LayoutWrapper>
            }
          />

          {/* 인증이 필요한 라우트는 여기에 추가 */}
        </Route>
        <Route path="*" element={<Navigate to="/dashboard" />} />
      </Routes>
    </Suspense>
  );
}

export default App;
