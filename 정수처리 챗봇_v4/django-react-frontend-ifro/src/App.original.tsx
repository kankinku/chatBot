import React from "react";
import { Routes, Route, Navigate, Outlet } from "react-router-dom";
import Dashboard from "./features/dashboard/components/index";
import { AdminDashboard } from "./features/dashboard";
import LoginForm from "./features/auth/components/LoginForm";
import RegisterForm from "./features/auth/components/RegisterForm";
import Settings from "./shared/components/Settings";
import {
  ProposalList,
  CreateProposalForm,
  EditProposalForm,
  ProposalDetail,
  AdminProposalManagement,
} from "./features/policy-proposals";
import { MainLayout } from "./shared/components/layout";

const PrivateRoute = () => {
  const isAuth = !!localStorage.getItem("access");
  return isAuth ? <Outlet /> : <Navigate to="/login" />;
};

const LayoutWrapper = ({ children }: { children: React.ReactNode }) => {
  return <MainLayout>{children}</MainLayout>;
};

function App() {
  return (
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
  );
}

export default App;
