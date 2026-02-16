const Footer = () => {
  return (
    <div
      className="relative w-full h-[30vh]"
      style={{
        clipPath: "polygon(0 0, 100% 0, 100% 100%, 0% 100%)",
      }}
    >
      <footer className="footer h-[30vh] sm:footer-horizontal bg-base-100 text-base-content items-center justify-center p-4 fixed bottom-0">
        <div className="bottom flex flex-col justify-center items-center gap-2 ">
          <p>
            &copy; {new Date().getFullYear()} TechniqueAI - Designed & Developed
            by{" "}
            <a
              href="https://thony-enechi.onrender.com"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-block underline text-info transition-transform duration-200 hover:scale-95"
            >
              Thony.
            </a>
          </p>
        </div>
      </footer>
    </div>
  );
};

export default Footer;
